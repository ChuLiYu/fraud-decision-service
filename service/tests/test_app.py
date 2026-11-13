import time
from concurrent.futures import ThreadPoolExecutor

from fraud_service.app import create_app
from fraud_service.model import Policy

LOW = {
    "transaction_id": 1,
    "timestamp_seconds": 90_000,
    "amount": "12.50",
    "fields": {"ProductCD": "W", "D1": 300, "DeviceType": "desktop"},
}
HIGH = {
    "transaction_id": 2,
    "timestamp_seconds": 90_000,
    "amount": "950.00",
    "fields": {"ProductCD": "C", "D1": 1, "DeviceType": "mobile"},
}


def test_health_names_the_demo_model(client):
    body = client.get("/health").json()
    assert body["model_kind"] == "demo" and body["model_version"].startswith("demo-")


def test_low_and_high_risk_get_different_outcomes(client):
    low = client.post("/v1/decisions", json=LOW).json()
    high = client.post("/v1/decisions", json=HIGH).json()
    assert low["outcome"] == "approve" and 0 <= low["score"] <= 1
    assert high["outcome"] in ("review", "decline") and high["score"] > low["score"]


def test_policy_thresholds():
    policy = Policy("p", "v", t_review=0.2, t_decline=0.5)
    assert policy.outcome(0.1)[0] == "approve"
    assert policy.outcome(0.2)[0] == "review"
    assert policy.outcome(0.5)[0] == "decline"
    assert Policy("p", "v", t_review=0.2, t_decline=None).outcome(0.99)[0] == "review"


def test_invalid_requests_are_rejected(client):
    for bad in (
        {**LOW, "amount": 12.5},  # float, not a decimal string
        {**LOW, "amount": "-1"},
        {**LOW, "amount": "1.23456"},
        {**LOW, "extra": 1},
        {**LOW, "fields": {"not_a_field": 1}},
        {**LOW, "fields": {"isFraud": 1}},  # the label is never an input
    ):
        assert client.post("/v1/decisions", json=bad).status_code == 422, bad
    assert client.post("/v1/decisions", json={**LOW, "amount": "0"}).status_code == 422


def test_model_error_routes_to_review_never_approve(make_client):
    def broken(*_):
        raise RuntimeError("model crashed")

    body = make_client(scorer=broken).post("/v1/decisions", json=LOW).json()
    assert (body["outcome"], body["reason"], body["score"]) == ("review", "model_error", None)


def test_invalid_score_routes_to_review(make_client):
    for bad in (float("nan"), 1.5, -0.1):
        body = make_client(scorer=lambda *_, v=bad: v).post("/v1/decisions", json=LOW).json()
        assert (body["outcome"], body["reason"], body["score"]) == ("review", "invalid_score", None)


def test_timeout_routes_to_review(make_client):
    def slow(*_):
        time.sleep(0.3)
        return 0.0

    body = make_client(scorer=slow, timeout_ms=50).post("/v1/decisions", json=LOW).json()
    assert (body["outcome"], body["reason"]) == ("review", "model_timeout")


def test_same_key_same_body_replays_the_decision(client):
    headers = {"Idempotency-Key": "k1"}
    first = client.post("/v1/decisions", json=LOW, headers=headers)
    second = client.post("/v1/decisions", json=LOW, headers=headers)
    assert second.json() == first.json()
    assert second.headers.get("Idempotent-Replayed") == "true"


def test_same_key_different_body_is_a_conflict(client):
    headers = {"Idempotency-Key": "k2"}
    assert client.post("/v1/decisions", json=LOW, headers=headers).status_code == 200
    assert client.post("/v1/decisions", json=HIGH, headers=headers).status_code == 409


def test_concurrent_duplicates_store_one_decision(bundle_dir, tmp_path):
    from fastapi.testclient import TestClient

    db = tmp_path / "race.db"
    clients = [TestClient(create_app(bundle_dir, db)) for _ in range(8)]
    headers = {"Idempotency-Key": "race"}
    with ThreadPoolExecutor(8) as pool:
        bodies = list(
            pool.map(lambda c: c.post("/v1/decisions", json=LOW, headers=headers).json(), clients)
        )
    assert len({b["decision_id"] for b in bodies}) == 1
    from fraud_service.store import Store

    assert Store(db).count() == 1
