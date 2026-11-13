"""Decision log and idempotency keys in SQLite.

Every decision is recorded with the request hash, model and policy versions,
outcome, reason, and latency, so it can be explained later. An
``Idempotency-Key`` is unique in the table: concurrent duplicates are settled
by the database constraint, not by in-process locks.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    decision_id     TEXT PRIMARY KEY,
    idempotency_key TEXT UNIQUE,
    request_hash    TEXT NOT NULL,
    transaction_id  INTEGER NOT NULL,
    amount          TEXT NOT NULL,   -- decimal string, never a binary float
    outcome         TEXT NOT NULL CHECK (outcome IN ('approve', 'review', 'decline')),
    reason          TEXT NOT NULL,
    score           REAL,            -- NULL when no valid score was obtained
    model_version   TEXT NOT NULL,
    policy_version  TEXT NOT NULL,
    latency_ms      REAL NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
)
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
    "decision_id, transaction_id, outcome, reason, score, model_version, policy_version, "
    "request_hash"
)


class Store:
    """Thin wrapper over one SQLite file; a connection per call keeps threads apart."""

    def __init__(self, path: Path) -> None:
        """Create the database file and table if needed."""
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=5.0)

    def find(self, key: str, request_hash: str) -> Decision | None:
        """Stored decision for ``key``, if any.

        Raises:
            IdempotencyConflict: If ``key`` was used with a different body.
        """
        with self._connect() as db:
            row = db.execute(
                f"SELECT {_COLUMNS} FROM decisions WHERE idempotency_key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        decision = Decision(*row)
        if decision.request_hash != request_hash:
            raise IdempotencyConflict(key)
        return decision

    def save(self, decision: Decision, key: str | None, amount: str, latency_ms: float) -> Decision:
        """Record ``decision``; if ``key`` won a race elsewhere, return that one instead.

        Raises:
            IdempotencyConflict: If a concurrent request used ``key`` with another body.
        """
        try:
            with self._connect() as db:
                db.execute(
                    "INSERT INTO decisions (decision_id, idempotency_key, request_hash, "
                    "transaction_id, amount, outcome, reason, score, model_version, "
                    "policy_version, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
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
                )
        except sqlite3.IntegrityError:
            if key is None:
                raise
            stored = self.find(key, decision.request_hash)
            if stored is None:
                raise
            return stored
        return decision

    def count(self) -> int:
        """Number of stored decisions."""
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
