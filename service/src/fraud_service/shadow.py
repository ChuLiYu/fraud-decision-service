"""Kafka consumer that scores decisions with a shadow bundle off the request path.

Delivery is at least once.  A PostgreSQL primary key on ``event_id`` makes a
redelivery a no-op, and the Kafka offset is committed only after the shadow
comparison transaction commits.  Shadow failures are recorded for review but
never reach the active decision path (ADR-003).

Run: ``python -m fraud_service.shadow``.  ``DATABASE_URL``,
``KAFKA_BOOTSTRAP_SERVERS``, and ``SHADOW_BUNDLE_DIR`` are required.
``EVENTS_TOPIC`` defaults to ``fraud.decision-events`` and ``CONSUMER_GROUP``
defaults to ``shadow-scoring``.
"""

from __future__ import annotations

import logging
import math
import os
import re
import signal
import time
import uuid
from decimal import Decimal, InvalidOperation
from numbers import Real
from pathlib import Path
from typing import Protocol

from confluent_kafka import Consumer, KafkaError, KafkaException, Message
from psycopg import Connection

from fraud_service.consumer import InvalidEvent, Record, parse
from fraud_service.model import Policy, load
from fraud_service.store import create_schema

log = logging.getLogger("fraud_service.shadow")

DEFAULT_TOPIC = "fraud.decision-events"
DEFAULT_GROUP = "shadow-scoring"
_AMOUNT = re.compile(r"^\d{1,9}(?:\.\d{1,3})?$")
_OUTCOMES = {"approve", "review", "decline"}
_REQUIRED_FIELDS = {
    "event_id",
    "event_type",
    "schema_version",
    "occurred_at",
    "decision_id",
    "transaction_id",
    "amount",
    "outcome",
    "reason",
    "score",
    "model_version",
    "policy_version",
    "latency_ms",
}
_ALLOWED_FIELDS = _REQUIRED_FIELDS | {"request"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS shadow_scores (
    event_id             UUID PRIMARY KEY,
    decision_id          UUID NOT NULL,
    transaction_id       BIGINT NOT NULL,
    active_model_version TEXT NOT NULL,
    active_score         DOUBLE PRECISION,
    active_outcome       TEXT NOT NULL CHECK (active_outcome IN ('approve', 'review', 'decline')),
    shadow_model_version TEXT NOT NULL,
    shadow_policy_version TEXT NOT NULL,
    shadow_score         DOUBLE PRECISION,
    shadow_outcome       TEXT CHECK (shadow_outcome IN ('approve', 'review', 'decline')),
    status               TEXT NOT NULL CHECK (
        status IN ('scored', 'shadow_error', 'invalid_score', 'no_request')
    ),
    error                TEXT CHECK (error IS NULL OR char_length(error) <= 500),
    shadow_latency_ms    DOUBLE PRECISION NOT NULL CHECK (shadow_latency_ms >= 0),
    created_at           TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class ShadowBundle(Protocol):
    """The serving-bundle operations used by the shadow consumer."""

    model_version: str
    policy: Policy

    def score(self, transaction_id: int, seconds: int, amount: float, fields: dict) -> float:
        """Return one model score."""


def _invalid(message: str) -> None:
    raise InvalidEvent(message)


def _is_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _validate_uuid(event: dict, name: str) -> None:
    value = event[name]
    if not isinstance(value, str):
        _invalid(f"{name} is not a UUID")
    try:
        uuid.UUID(value)
    except ValueError as exc:
        raise InvalidEvent(f"{name} is not a UUID") from exc


def _validate_amount(value: object, name: str) -> Decimal:
    if not isinstance(value, str) or _AMOUNT.fullmatch(value) is None:
        _invalid(f"{name} is not a decimal amount")
    try:
        return Decimal(value)
    except InvalidOperation as exc:  # defensive; the pattern should exclude this
        raise InvalidEvent(f"{name} is not a decimal amount") from exc


def _event(record: Record) -> dict:
    """Parse and validate the full v1 payload needed for shadow scoring."""
    event = parse(record.value)
    missing = _REQUIRED_FIELDS - event.keys()
    if missing:
        _invalid(f"missing fields: {sorted(missing)}")
    unknown = event.keys() - _ALLOWED_FIELDS
    if unknown:
        _invalid(f"unknown fields: {sorted(unknown)}")
    _validate_uuid(event, "event_id")
    _validate_uuid(event, "decision_id")
    if not isinstance(event["transaction_id"], int) or isinstance(event["transaction_id"], bool):
        _invalid("transaction_id is not an integer")
    if event["transaction_id"] < 0:
        _invalid("transaction_id is negative")
    _validate_amount(event["amount"], "amount")
    if not isinstance(event["outcome"], str) or event["outcome"] not in _OUTCOMES:
        _invalid("outcome is unsupported")
    if not isinstance(event["model_version"], str) or not isinstance(event["policy_version"], str):
        _invalid("model and policy versions must be strings")
    if not isinstance(event["reason"], str) or not isinstance(event["occurred_at"], str):
        _invalid("reason and occurred_at must be strings")
    if not _is_number(event["latency_ms"]) or not math.isfinite(event["latency_ms"]):
        _invalid("latency_ms is not finite")
    if event["latency_ms"] < 0:
        _invalid("latency_ms is negative")
    active_score = event["score"]
    if active_score is not None:
        if not _is_number(active_score) or not math.isfinite(active_score):
            _invalid("score is not finite")
        if not 0.0 <= active_score <= 1.0:
            _invalid("score is outside [0, 1]")

    request = event.get("request")
    if request is None:
        return event
    if not isinstance(request, dict):
        _invalid("request is not an object")
    if set(request) != {"timestamp_seconds", "amount", "fields"}:
        _invalid("request fields are invalid")
    if not isinstance(request["timestamp_seconds"], int) or isinstance(
        request["timestamp_seconds"], bool
    ):
        _invalid("request.timestamp_seconds is not an integer")
    if request["timestamp_seconds"] < 0:
        _invalid("request.timestamp_seconds is negative")
    _validate_amount(request["amount"], "request.amount")
    fields = request["fields"]
    if not isinstance(fields, dict):
        _invalid("request.fields is not an object")
    for value in fields.values():
        if value is not None and not isinstance(value, str) and not _is_number(value):
            _invalid("request.fields contains an unsupported value")
        if _is_number(value) and not math.isfinite(value):
            _invalid("request.fields contains a non-finite number")
    return event


def _error(exc: Exception) -> str:
    """Store a bounded, useful error without allowing a verbose exception to break SQL."""
    return f"{type(exc).__name__}: {exc}"[:500]


def _score(
    bundle: ShadowBundle, event: dict
) -> tuple[float | None, str | None, str, str | None, float]:
    """Score one valid request and turn all model failures into an auditable status."""
    request = event.get("request")
    if request is None:
        return None, None, "no_request", None, 0.0

    started = time.perf_counter()
    try:
        score = bundle.score(
            event["transaction_id"],
            request["timestamp_seconds"],
            float(Decimal(request["amount"])),
            request["fields"],
        )
        latency_ms = (time.perf_counter() - started) * 1_000
        if not _is_number(score) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
            return None, None, "invalid_score", f"invalid score: {score!r}", latency_ms
        outcome, _ = bundle.policy.outcome(float(score))
        if not isinstance(outcome, str) or outcome not in _OUTCOMES:
            return None, None, "invalid_score", f"invalid outcome: {outcome!r}", latency_ms
        return float(score), outcome, "scored", None, latency_ms
    except Exception as exc:  # ADR-003: failures are recorded, never sent to active serving.
        latency_ms = (time.perf_counter() - started) * 1_000
        error = _error(exc)
        log.exception("shadow scoring failed for event %s: %s", event["event_id"], error)
        return None, None, "shadow_error", error, latency_ms


def apply(conn: Connection, bundle: ShadowBundle, record: Record) -> bool:
    """Store one active/shadow comparison, or return ``False`` for a duplicate event.

    Raises:
        InvalidEvent: If the payload is not a complete supported decision event.
    """
    event = _event(record)
    with conn.transaction():
        existing = conn.execute(
            "SELECT 1 FROM shadow_scores WHERE event_id = %s", (event["event_id"],)
        ).fetchone()
        if existing is not None:
            return False
        shadow_score, shadow_outcome, status, error, latency_ms = _score(bundle, event)
        inserted = conn.execute(
            "INSERT INTO shadow_scores (event_id, decision_id, transaction_id, "
            "active_model_version, active_score, active_outcome, shadow_model_version, "
            "shadow_policy_version, shadow_score, "
            "shadow_outcome, status, error, shadow_latency_ms) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (event_id) DO NOTHING RETURNING event_id",
            (
                event["event_id"],
                event["decision_id"],
                event["transaction_id"],
                event["model_version"],
                event["score"],
                event["outcome"],
                bundle.model_version,
                bundle.policy.version,
                shadow_score,
                shadow_outcome,
                status,
                error,
                latency_ms,
            ),
        ).fetchone()
    return inserted is not None


def _record(message: Message) -> Record:
    return Record(message.topic(), message.partition(), message.offset(), message.value())


def run(database_url: str, bootstrap: str, bundle_dir: Path, topic: str, group: str) -> None:
    """Consume shadow events until SIGTERM/SIGINT, committing offsets after PostgreSQL."""
    stopping = False

    def stop(*_: object) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    bundle = load(bundle_dir)
    conn = Connection.connect(database_url, autocommit=True)
    create_schema(conn, SCHEMA)
    consumer = Consumer(
        {
            "bootstrap.servers": bootstrap,
            "group.id": group,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
            "topic.metadata.refresh.interval.ms": 5000,
        }
    )
    consumer.subscribe([topic])
    log.info("shadow-scoring %s as group %s with model %s", topic, group, bundle.model_version)
    try:
        while not stopping:
            message = consumer.poll(1.0)
            if message is None:
                continue
            if message.error():
                if message.error().code() == KafkaError.UNKNOWN_TOPIC_OR_PART:
                    log.info("topic %s does not exist yet; waiting", topic)
                    continue
                raise KafkaException(message.error())
            record = _record(message)
            applied = apply(conn, bundle, record)
            consumer.commit(message=message, asynchronous=False)
            log.info(
                "%s %s@%d:%d",
                "shadow-scored" if applied else "duplicate skipped",
                record.topic,
                record.partition,
                record.offset,
            )
    finally:
        consumer.close()
        conn.close()


def main() -> None:
    """Entry point for ``python -m fraud_service.shadow``."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run(
        os.environ["DATABASE_URL"],
        os.environ["KAFKA_BOOTSTRAP_SERVERS"],
        Path(os.environ["SHADOW_BUNDLE_DIR"]),
        os.environ.get("EVENTS_TOPIC", DEFAULT_TOPIC),
        os.environ.get("CONSUMER_GROUP", DEFAULT_GROUP),
    )


if __name__ == "__main__":
    main()
