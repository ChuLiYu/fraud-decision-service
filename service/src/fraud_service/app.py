"""HTTP API: one card-not-present transaction in, approve / review / decline out.

Failure rule (ADR-003): if no valid score is obtained in time (model error,
timeout, invalid score), the outcome is ``review`` with the cause as reason,
never ``approve``. The failure is logged, and the decision records a null
score so it is visible in the decision log.

Run: ``uvicorn fraud_service.app:create_app --factory``. Settings come from
environment variables: ``BUNDLE_DIR`` (required), ``DATABASE_URL``
(required, PostgreSQL), ``SCORE_TIMEOUT_MS`` (default 500).

Each new decision is written together with its ``decision.made`` event in
one database transaction (outbox, ADR-006); Debezium publishes it to Kafka.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from decimal import Decimal
from numbers import Real
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fraud_service import model
from fraud_service.store import Decision, IdempotencyConflict, Store

log = logging.getLogger("fraud_service")

CONFLICT = "Idempotency-Key reused with a different body"
Scorer = Callable[[int, int, float, dict], float]


class DecisionRequest(BaseModel):
    """One transaction. ``fields`` holds optional raw IEEE-CIS fields by name."""

    model_config = ConfigDict(extra="forbid")

    transaction_id: int = Field(ge=0)
    timestamp_seconds: int = Field(ge=0, description="Seconds on the dataset's time axis")
    amount: str = Field(
        pattern=r"^\d{1,9}(\.\d{1,3})?$", description="Decimal string, e.g. '49.99'"
    )
    fields: dict[str, str | int | float | None] = Field(default_factory=dict)

    @field_validator("transaction_id", "timestamp_seconds", mode="before")
    @classmethod
    def finite_top_level_numbers(cls, value: object) -> object:
        """Reject JSON non-finite number extensions before integer parsing."""
        if isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(value):
            raise ValueError("must be finite")
        return value

    @field_validator("fields")
    @classmethod
    def finite_field_numbers(cls, fields: dict[str, str | int | float | None]) -> dict:
        """Keep JSONB events valid by rejecting NaN and infinities at the API boundary."""
        if any(
            isinstance(value, Real) and not isinstance(value, bool) and not math.isfinite(value)
            for value in fields.values()
        ):
            raise ValueError("fields must not contain non-finite numbers")
        return fields


class DecisionResponse(BaseModel):
    """The decision and what produced it."""

    decision_id: str
    transaction_id: int
    outcome: str
    reason: str
    score: float | None
    model_version: str
    policy_version: str


def _hash(body: DecisionRequest) -> str:
    canonical = json.dumps(body.model_dump(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def create_app(
    bundle_dir: Path | None = None,
    database_url: str | None = None,
    timeout_ms: float | None = None,
    scorer: Scorer | None = None,
) -> FastAPI:
    """Build the app; arguments override environment variables (tests use them)."""
    bundle_dir = bundle_dir or Path(os.environ["BUNDLE_DIR"])
    database_url = database_url or os.environ["DATABASE_URL"]
    timeout_s = (timeout_ms or float(os.environ.get("SCORE_TIMEOUT_MS", "500"))) / 1000

    bundle = model.load(bundle_dir)
    store = Store(database_url)
    score_fn: Scorer = scorer or bundle.score
    allowed = set(bundle.request_fields)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        store.close()

    app = FastAPI(title="Fraud decision service", version="0.2.0", lifespan=lifespan)
    app.state.store = store

    @app.get("/health")
    def health() -> dict:
        """Liveness plus which model and policy are loaded."""
        return {
            "status": "ok",
            "model_kind": bundle.kind,
            "model_version": bundle.model_version,
            "policy_version": bundle.policy.version,
        }

    @app.post("/v1/decisions", response_model=DecisionResponse)
    async def decide(
        body: DecisionRequest,
        idempotency_key: Annotated[str | None, Header(max_length=128)] = None,
    ) -> JSONResponse:
        """Decide one transaction; replays the stored decision for a repeated key."""
        unknown = sorted(set(body.fields) - allowed)
        if unknown:
            raise HTTPException(422, detail={"unknown_fields": unknown[:20]})
        if Decimal(body.amount) <= 0:
            raise HTTPException(422, detail="amount must be positive")

        request_hash = _hash(body)
        if idempotency_key is not None:
            try:
                stored = store.find(idempotency_key, request_hash)
            except IdempotencyConflict:
                raise HTTPException(409, detail=CONFLICT) from None
            if stored is not None:
                return _respond(stored, replayed=True)

        started = time.perf_counter()
        score: float | None = None
        try:
            score = await asyncio.wait_for(
                asyncio.to_thread(
                    score_fn,
                    body.transaction_id,
                    body.timestamp_seconds,
                    float(body.amount),
                    body.fields,
                ),
                timeout_s,
            )
            if not (isinstance(score, float) and 0.0 <= score <= 1.0):  # NaN fails too
                log.error("invalid score %r for transaction %s", score, body.transaction_id)
                score, outcome, reason = None, "review", "invalid_score"
            else:
                outcome, reason = bundle.policy.outcome(score)
        except TimeoutError:
            log.warning("scoring timed out for transaction %s", body.transaction_id)
            score, outcome, reason = None, "review", "model_timeout"
        except Exception:
            log.exception("scoring failed for transaction %s", body.transaction_id)
            score, outcome, reason = None, "review", "model_error"
        latency_ms = (time.perf_counter() - started) * 1000

        decision = Decision(
            decision_id=str(uuid.uuid4()),
            transaction_id=body.transaction_id,
            outcome=outcome,
            reason=reason,
            score=score,
            model_version=bundle.model_version,
            policy_version=bundle.policy.version,
            request_hash=request_hash,
        )
        try:
            saved = store.save(
                decision,
                idempotency_key,
                body.timestamp_seconds,
                body.amount,
                body.fields,
                latency_ms,
            )
        except IdempotencyConflict:
            raise HTTPException(409, detail=CONFLICT) from None
        return _respond(saved, replayed=saved.decision_id != decision.decision_id)

    return app


def _respond(decision: Decision, replayed: bool) -> JSONResponse:
    body = DecisionResponse(
        decision_id=decision.decision_id,
        transaction_id=decision.transaction_id,
        outcome=decision.outcome,
        reason=decision.reason,
        score=decision.score,
        model_version=decision.model_version,
        policy_version=decision.policy_version,
    )
    headers = {"Idempotent-Replayed": "true"} if replayed else {}
    return JSONResponse(body.model_dump(), headers=headers)
