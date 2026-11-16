"""Decision log, idempotency keys, and the event outbox in PostgreSQL.

Every decision is recorded with the request hash, model and policy versions,
outcome, reason, and latency, so it can be explained later. An
``Idempotency-Key`` is unique in the table: concurrent duplicates are settled
by the database constraint, not by in-process locks.

A new decision and its ``decision.made`` event are inserted in one
transaction (transactional outbox, ADR-006). Debezium reads the ``outbox``
table from the write-ahead log and publishes the payload to Kafka; this
module never talks to Kafka. A replayed or conflicting request inserts
neither row, so every event corresponds to exactly one stored decision.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from psycopg import Connection
from psycopg.types.json import Jsonb
from psycopg_pool import ConnectionPool

EVENT_TYPE = "decision.made"
EVENT_SCHEMA_VERSION = 1
AGGREGATE_TYPE = "decision"
_SCHEMA_LOCK = 7_301_001  # advisory lock id: serialise CREATE TABLE across processes

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id     UUID PRIMARY KEY,
    idempotency_key TEXT UNIQUE,
    request_hash    TEXT NOT NULL,
    transaction_id  BIGINT NOT NULL,
    amount          NUMERIC(12, 3) NOT NULL,   -- decimal, never a binary float
    outcome         TEXT NOT NULL CHECK (outcome IN ('approve', 'review', 'decline')),
    reason          TEXT NOT NULL,
    score           DOUBLE PRECISION,          -- NULL when no valid score was obtained
    model_version   TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    latency_ms      DOUBLE PRECISION NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
-- Column names follow the Debezium Outbox Event Router defaults.
CREATE TABLE IF NOT EXISTS outbox (
    id            UUID PRIMARY KEY,            -- event_id
    aggregatetype TEXT NOT NULL,
    aggregateid   TEXT NOT NULL,               -- decision_id; becomes the Kafka key
    type          TEXT NOT NULL,
    payload       JSONB NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


class IdempotencyConflict(Exception):
    """The key was already used with a different request body."""


@dataclass(frozen=True)
class Decision:
    """One stored decision."""

    decision_id: str
    transaction_id: int
    outcome: str
    reason: str
    score: float | None
    model_version: str
    policy_version: str
    request_hash: str


_COLUMNS = (
    "decision_id::text, transaction_id, outcome, reason, score, model_version, "
    "policy_version, request_hash"
)


def _iso_utc(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat().replace("+00:00", "Z")


def event_payload(
    event_id: str,
    decision: Decision,
    timestamp_seconds: int,
    amount: str,
    fields: dict[str, str | int | float | None],
    latency_ms: float,
    occurred_at: datetime,
) -> dict:
    """The ``decision.made`` v1 payload (docs/contracts/events/decision-made.v1.schema.json)."""
    return {
        "event_id": event_id,
        "event_type": EVENT_TYPE,
        "schema_version": EVENT_SCHEMA_VERSION,
        "occurred_at": _iso_utc(occurred_at),
        "decision_id": decision.decision_id,
        "transaction_id": decision.transaction_id,
        "amount": amount,
        "request": {
            "timestamp_seconds": timestamp_seconds,
            "amount": amount,
            "fields": fields,
        },
        "outcome": decision.outcome,
        "reason": decision.reason,
        "score": decision.score,
        "model_version": decision.model_version,
        "policy_version": decision.policy_version,
        "latency_ms": round(latency_ms, 3),
    }


def create_schema(conn: Connection, ddl: str) -> None:
    """Run ``ddl`` once even when several processes start together."""
    with conn.transaction():
        conn.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK,))
        conn.execute(ddl)


class Store:
    """Decision log over a connection pool; safe to share between threads."""

    def __init__(self, database_url: str, max_size: int = 10) -> None:
        """Connect and create the tables if needed."""
        self.pool = ConnectionPool(
            database_url, min_size=1, max_size=max_size, open=True, kwargs={"autocommit": True}
        )
        with self.pool.connection() as conn:
            create_schema(conn, SCHEMA)

    def close(self) -> None:
        """Close all pooled connections."""
        self.pool.close()

    def find(self, key: str, request_hash: str) -> Decision | None:
        """Stored decision for ``key``, if any.

        Raises:
            IdempotencyConflict: If ``key`` was used with a different body.
        """
        with self.pool.connection() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM decisions WHERE idempotency_key = %s", (key,)
            ).fetchone()
        if row is None:
            return None
        decision = Decision(*row)
        if decision.request_hash != request_hash:
            raise IdempotencyConflict(key)
        return decision

    def save(
        self,
        decision: Decision,
        key: str | None,
        timestamp_seconds: int,
        amount: str,
        fields: dict[str, str | int | float | None],
        latency_ms: float,
    ) -> Decision:
        """Record ``decision`` and its event; if ``key`` won a race elsewhere, return that one.

        Raises:
            IdempotencyConflict: If a concurrent request used ``key`` with another body.
        """
        with self.pool.connection() as conn, conn.transaction():
            row = conn.execute(
                "INSERT INTO decisions (decision_id, idempotency_key, request_hash, "
                "transaction_id, amount, outcome, reason, score, model_version, "
                "policy_version, latency_ms) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (idempotency_key) DO NOTHING RETURNING created_at",
                (
                    decision.decision_id,
                    key,
                    decision.request_hash,
                    decision.transaction_id,
                    amount,
                    decision.outcome,
                    decision.reason,
                    decision.score,
                    decision.model_version,
                    decision.policy_version,
                    latency_ms,
                ),
            ).fetchone()
            if row is not None:
                event_id = str(uuid.uuid4())
                payload = event_payload(
                    event_id,
                    decision,
                    timestamp_seconds,
                    amount,
                    fields,
                    latency_ms,
                    row[0],
                )
                conn.execute(
                    "INSERT INTO outbox (id, aggregatetype, aggregateid, type, payload) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (event_id, AGGREGATE_TYPE, decision.decision_id, EVENT_TYPE, Jsonb(payload)),
                )
                return decision
        # The key exists already (possibly committed by a concurrent request just now).
        if key is None:
            raise RuntimeError("decision insert skipped without an idempotency key")
        stored = self.find(key, decision.request_hash)
        if stored is None:
            raise RuntimeError(f"idempotency key {key!r} conflicted but no row was found")
        return stored

    def count(self, table: str = "decisions") -> int:
        """Number of rows in ``decisions`` or ``outbox``."""
        if table not in ("decisions", "outbox"):
            raise ValueError(table)
        with self.pool.connection() as conn:
            return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    def outbox_payloads(self) -> list[dict]:
        """All event payloads in insertion order (tests and the kill-test script)."""
        with self.pool.connection() as conn:
            rows = conn.execute("SELECT payload FROM outbox ORDER BY created_at, id").fetchall()
        return [r[0] for r in rows]
