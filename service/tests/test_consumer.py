"""Consumer projection: duplicates and restarts change counts once per event."""

import json
import uuid

import psycopg
import pytest

from fraud_service.consumer import SCHEMA, InvalidEvent, Record, apply
from fraud_service.store import create_schema


def event(outcome="approve", model="demo-1", event_id=None) -> dict:
    return {
        "event_id": event_id or str(uuid.uuid4()),
        "event_type": "decision.made",
        "schema_version": 1,
        "decision_id": str(uuid.uuid4()),
        "model_version": model,
        "outcome": outcome,
    }


def record(payload, offset=0) -> Record:
    value = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return Record("fraud.decision-events", 0, offset, value)


@pytest.fixture
def conn(database_url):
    with psycopg.connect(database_url, autocommit=True) as c:
        create_schema(c, SCHEMA)
        yield c


def counts(conn) -> dict:
    rows = conn.execute("SELECT model_version, outcome, n FROM outcome_counts").fetchall()
    return {(m, o): n for m, o, n in rows}


def test_events_are_counted_by_model_and_outcome(conn):
    for e in (event("approve"), event("approve"), event("review"), event("approve", "demo-2")):
        assert apply(conn, record(e)) is True
    assert counts(conn) == {
        ("demo-1", "approve"): 2,
        ("demo-1", "review"): 1,
        ("demo-2", "approve"): 1,
    }


def test_redelivered_event_is_counted_once(conn):
    e = event("decline")
    assert apply(conn, record(e, offset=5)) is True
    # Same event again, e.g. after a crash before the offset commit or a CDC replay.
    assert apply(conn, record(e, offset=5)) is False
    assert apply(conn, record(e, offset=9)) is False
    assert counts(conn) == {("demo-1", "decline"): 1}
    assert conn.execute("SELECT COUNT(*) FROM consumed_events").fetchone()[0] == 1


def test_restart_with_a_new_connection_keeps_dedupe(conn, database_url):
    e = event()
    apply(conn, record(e))
    with psycopg.connect(database_url, autocommit=True) as restarted:
        assert apply(restarted, record(e)) is False
    assert counts(conn) == {("demo-1", "approve"): 1}


@pytest.mark.parametrize(
    "bad",
    [
        b"not json",
        b"null",  # tombstone-like value
        {**event(), "schema_version": 2},
        {**event(), "event_type": "decision.deleted"},
        {k: v for k, v in event().items() if k != "event_id"},
    ],
)
def test_invalid_events_fail_loudly_and_write_nothing(conn, bad):
    with pytest.raises(InvalidEvent):
        apply(conn, record(bad))
    assert counts(conn) == {}
