"""Asynchronous shadow scoring records comparisons without changing active decisions."""

import json
import uuid
from types import SimpleNamespace

import psycopg
import pytest

from fraud_service.consumer import InvalidEvent, Record
from fraud_service.model import load
from fraud_service.shadow import SCHEMA, apply
from fraud_service.shadow_report import markdown, summary
from fraud_service.store import create_schema


def event(**overrides) -> dict:
    """A complete current decision event with an optional request payload."""
    payload = {
        "event_id": str(uuid.uuid4()),
        "event_type": "decision.made",
        "schema_version": 1,
        "occurred_at": "2026-11-15T12:00:00Z",
        "decision_id": str(uuid.uuid4()),
        "transaction_id": 12,
        "amount": "12.500",
        "request": {
            "timestamp_seconds": 100_000,
            "amount": "12.500",
            "fields": {"D1": 12.0, "ProductCD": "C"},
        },
        "outcome": "approve",
        "reason": "score_below_review_threshold",
        "score": 0.1,
        "model_version": "active-1",
        "policy_version": "active-policy-1",
        "latency_ms": 1.0,
    }
    payload.update(overrides)
    return payload


def record(payload, offset=0) -> Record:
    """Turn an event object into the consumer-facing record value."""
    value = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    return Record("fraud.decision-events", 0, offset, value)


@pytest.fixture
def conn(database_url):
    with psycopg.connect(database_url, autocommit=True) as connection:
        create_schema(connection, SCHEMA)
        yield connection


@pytest.fixture
def shadow_bundle(shadow_bundle_dir):
    return load(shadow_bundle_dir)


def row(conn):
    return conn.execute(
        "SELECT event_id::text, active_model_version, shadow_model_version, shadow_outcome, "
        "status, error FROM shadow_scores"
    ).fetchone()


def fake_bundle(score):
    """A minimal bundle that makes score failure paths deterministic."""
    return SimpleNamespace(
        model_version="shadow-1",
        policy=SimpleNamespace(
            version="shadow-policy-1",
            outcome=lambda value: ("review", "test") if value >= 0.3 else ("approve", "test"),
        ),
        score=score,
    )


def test_scored_row_has_both_model_versions_and_a_policy_outcome(conn, shadow_bundle):
    payload = event()

    assert apply(conn, shadow_bundle, record(payload)) is True

    stored = row(conn)
    assert stored[:3] == (payload["event_id"], "active-1", shadow_bundle.model_version)
    assert stored[3] in {"approve", "review", "decline"}
    assert stored[4:] == ("scored", None)
    assert conn.execute(
        "SELECT active_score, active_outcome, shadow_policy_version FROM shadow_scores"
    ).fetchone() == (0.1, "approve", shadow_bundle.policy.version)


def test_duplicate_event_is_not_scored_or_stored_twice(conn):
    payload = event()
    calls = 0

    def score(*args, **kwargs):
        nonlocal calls
        calls += 1
        return 0.2

    bundle = fake_bundle(score)
    assert apply(conn, bundle, record(payload)) is True
    assert apply(conn, bundle, record(payload, offset=9)) is False
    assert calls == 1
    assert conn.execute("SELECT COUNT(*) FROM shadow_scores").fetchone()[0] == 1


def test_scoring_exception_is_recorded_without_propagating(conn):
    def fail(*args, **kwargs):
        raise RuntimeError("shadow unavailable")

    assert apply(conn, fake_bundle(fail), record(event())) is True

    assert row(conn)[4:] == ("shadow_error", "RuntimeError: shadow unavailable")


@pytest.mark.parametrize("score", [float("nan"), 1.5])
def test_invalid_shadow_scores_are_recorded(conn, score):
    bundle = fake_bundle(lambda *args, **kwargs: score)

    assert apply(conn, bundle, record(event())) is True

    assert row(conn)[3:] == (None, "invalid_score", f"invalid score: {score!r}")


def test_event_without_request_is_recorded_for_comparison_completeness(conn, shadow_bundle):
    payload = event()
    payload.pop("request")

    assert apply(conn, shadow_bundle, record(payload)) is True

    assert row(conn)[3:] == (None, "no_request", None)


def test_event_with_unknown_top_level_properties_is_scored(conn, shadow_bundle):
    payload = event(compatible_addition="ignored by the shadow consumer")

    assert apply(conn, shadow_bundle, record(payload)) is True

    assert row(conn)[4:] == ("scored", None)


def test_event_with_unknown_request_properties_is_scored(conn, shadow_bundle):
    payload = event()
    payload["request"]["currency"] = "USD"

    assert apply(conn, shadow_bundle, record(payload)) is True

    assert row(conn)[4:] == ("scored", None)


def test_invalid_event_raises_without_a_shadow_row(conn, shadow_bundle):
    payload = event()
    payload.pop("decision_id")

    with pytest.raises(InvalidEvent):
        apply(conn, shadow_bundle, record(payload))

    assert conn.execute("SELECT COUNT(*) FROM shadow_scores").fetchone()[0] == 0


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda payload: payload["request"].pop("fields"),
            "missing request fields: ['fields']",
        ),
        (lambda payload: payload["request"].update(fields={"D1": []}), None),
        (lambda payload: payload["request"].update(timestamp_seconds=-1), None),
    ],
    ids=["missing-key", "unsupported-field-value", "negative-timestamp"],
)
def test_invalid_request_subobjects_raise_without_a_shadow_row(
    conn, shadow_bundle, mutate, message
):
    payload = event()
    mutate(payload)

    with pytest.raises(InvalidEvent) as error:
        apply(conn, shadow_bundle, record(payload))

    if message is not None:
        assert str(error.value) == message
    assert conn.execute("SELECT COUNT(*) FROM shadow_scores").fetchone()[0] == 0


def test_api_outbox_event_is_stored_with_the_shadow_bundle_version(conn, client, shadow_bundle):
    response = client.post(
        "/v1/decisions",
        json={
            "transaction_id": 42,
            "timestamp_seconds": 90_000,
            "amount": "12.50",
            "fields": {"ProductCD": "W", "D1": 300, "DeviceType": "desktop"},
        },
    )
    assert response.status_code == 200
    (payload,) = client.app.state.store.outbox_payloads()

    assert apply(conn, shadow_bundle, record(payload)) is True

    assert conn.execute(
        "SELECT status, shadow_model_version FROM shadow_scores WHERE event_id = %s",
        (payload["event_id"],),
    ).fetchone() == ("scored", shadow_bundle.model_version)


def test_summary_counts_agreement_confusion_and_score_difference(conn):
    rows = [
        event(event_id=str(uuid.uuid4()), outcome="approve", score=0.1),
        event(event_id=str(uuid.uuid4()), outcome="review", score=0.4),
        event(event_id=str(uuid.uuid4()), outcome="decline", score=0.9),
        event(event_id=str(uuid.uuid4()), outcome="review", score=None, request=None),
    ]
    scores = iter([0.2, 0.4, 0.7])
    bundle = fake_bundle(lambda *args, **kwargs: next(scores))
    for payload in rows:
        assert apply(conn, bundle, record(payload)) is True

    report = summary(conn)

    assert report["pairs"] == [
        {
            "active_model_version": "active-1",
            "shadow_model_version": "shadow-1",
            "status_counts": {
                "scored": 3,
                "shadow_error": 0,
                "invalid_score": 0,
                "no_request": 1,
            },
            "active_failures": 1,
            "outcome_agreement": {
                "matching": 2,
                "rows": 3,
                "rate": pytest.approx(2 / 3),
            },
            "confusion_matrix": {
                "approve": {"approve": 1},
                "review": {"review": 1},
                "decline": {"review": 1},
            },
            "score_difference": {
                "rows": 3,
                "mean_abs": pytest.approx(0.1),
                "max_abs": pytest.approx(0.2),
            },
        }
    ]
    assert (
        "Rows without an active score (excluded from agreement; may overlap shadow failures): 1"
        in markdown(report)
    )
