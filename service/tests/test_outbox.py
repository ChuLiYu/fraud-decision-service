"""Transactional outbox: one event per stored decision, matching the v1 contract."""

import json
import uuid
from datetime import datetime
from pathlib import Path

import jsonschema
import pytest
from test_app import HIGH, LOW

from fraud_service import store as store_module
from fraud_service.store import Decision, Store

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "docs/contracts/events/decision-made.v1.schema.json"
)
VALIDATOR = jsonschema.Draft202012Validator(
    json.loads(SCHEMA_PATH.read_text()), format_checker=jsonschema.FormatChecker()
)


def check_contract(payload: dict) -> None:
    VALIDATOR.validate(payload)
    datetime.fromisoformat(payload["occurred_at"])  # date-time format needs an extra package


def test_each_new_decision_writes_one_matching_event(client):
    requests = (LOW, HIGH)
    decisions = [client.post("/v1/decisions", json=body).json() for body in requests]
    payloads = client.app.state.store.outbox_payloads()
    assert len(payloads) == 2
    for request, decision, event in zip(requests, decisions, payloads, strict=True):
        check_contract(event)
        for field in ("decision_id", "transaction_id", "outcome", "reason", "score"):
            assert event[field] == decision[field]
        assert event["amount"] == request["amount"]
        assert event["request"] == {
            "timestamp_seconds": request["timestamp_seconds"],
            "amount": request["amount"],
            "fields": request["fields"],
        }


def test_failure_path_decisions_emit_events_with_null_score(make_client):
    def broken(*_):
        raise RuntimeError("model crashed")

    client = make_client(scorer=broken)
    client.post("/v1/decisions", json=LOW)
    (event,) = client.app.state.store.outbox_payloads()
    check_contract(event)
    assert (event["outcome"], event["reason"], event["score"]) == ("review", "model_error", None)
    assert event["request"] == {
        "timestamp_seconds": LOW["timestamp_seconds"],
        "amount": LOW["amount"],
        "fields": LOW["fields"],
    }


def test_contract_accepts_events_written_before_request_was_added():
    payload = {
        "event_id": str(uuid.uuid4()),
        "event_type": "decision.made",
        "schema_version": 1,
        "occurred_at": "2026-11-15T00:00:00Z",
        "decision_id": str(uuid.uuid4()),
        "transaction_id": LOW["transaction_id"],
        "amount": LOW["amount"],
        "outcome": "approve",
        "reason": "low_risk",
        "score": 0.1,
        "model_version": "test",
        "policy_version": "test",
        "latency_ms": 1.0,
    }

    check_contract(payload)


def test_contract_rejects_unknown_request_properties():
    payload = {
        "event_id": str(uuid.uuid4()),
        "event_type": "decision.made",
        "schema_version": 1,
        "occurred_at": "2026-11-15T00:00:00Z",
        "decision_id": str(uuid.uuid4()),
        "transaction_id": LOW["transaction_id"],
        "amount": LOW["amount"],
        "outcome": "approve",
        "reason": "low_risk",
        "score": 0.1,
        "model_version": "test",
        "policy_version": "test",
        "latency_ms": 1.0,
        "request": {
            "timestamp_seconds": LOW["timestamp_seconds"],
            "amount": LOW["amount"],
            "fields": LOW["fields"],
            "unexpected": "value",
        },
    }

    with pytest.raises(jsonschema.ValidationError):
        check_contract(payload)


def test_replay_and_conflict_write_no_event(client):
    headers = {"Idempotency-Key": "once"}
    client.post("/v1/decisions", json=LOW, headers=headers)
    assert client.post("/v1/decisions", json=LOW, headers=headers).status_code == 200
    assert client.post("/v1/decisions", json=HIGH, headers=headers).status_code == 409
    store = client.app.state.store
    assert store.count("decisions") == 1 and store.count("outbox") == 1


def test_failed_event_write_rolls_back_the_decision(database_url, monkeypatch):
    """No decision without its event: both rows commit together or not at all."""
    store = Store(database_url)

    def fail(*_):
        raise RuntimeError("event serialisation failed")

    monkeypatch.setattr(store_module, "event_payload", fail)
    try:
        decision = Decision(str(uuid.uuid4()), 1, "approve", "r", 0.1, "m", "p", "h")
        with pytest.raises(RuntimeError, match="event serialisation failed"):
            store.save(decision, "k", 90_000, "1.00", {"ProductCD": "W"}, 1.0)
        assert store.count("decisions") == 0 and store.count("outbox") == 0
    finally:
        store.close()
