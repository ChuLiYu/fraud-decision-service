"""A demo bundle built once from synthetic rows (no Competition Data)."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from fraud_ml.bundle import build, demo_params, synthetic_transactions
from fraud_ml.config import load_params

from fraud_service.app import create_app

ML_PARAMS = Path(__file__).resolve().parents[2] / "ml" / "params.yaml"


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory) -> Path:
    params = load_params(ML_PARAMS)
    params = demo_params(params)
    out = tmp_path_factory.mktemp("bundle")
    build(synthetic_transactions(n=5_000), params, out, "demo")
    return out


@pytest.fixture
def make_client(bundle_dir, tmp_path):
    def factory(**overrides) -> TestClient:
        return TestClient(create_app(bundle_dir, tmp_path / "decisions.db", **overrides))

    return factory


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
