"""Shared fixtures: a demo bundle from synthetic rows and a Postgres container.

The Postgres tests need Docker. Without it they fail (they are not skipped),
because the outbox and idempotency guarantees live in the database.
"""

from pathlib import Path

import psycopg
import pytest
from fastapi.testclient import TestClient
from fraud_ml.bundle import build, demo_params, synthetic_transactions
from fraud_ml.config import load_params
from testcontainers.community.postgres import PostgresContainer

from fraud_service.app import create_app

ML_PARAMS = Path(__file__).resolve().parents[2] / "ml" / "params.yaml"
POSTGRES_IMAGE = "postgres:18.7-alpine"  # same as compose.yaml
TABLES = ("decisions", "outbox", "consumed_events", "outcome_counts")


@pytest.fixture(scope="session")
def bundle_dir(tmp_path_factory) -> Path:
    params = load_params(ML_PARAMS)
    params = demo_params(params)
    out = tmp_path_factory.mktemp("bundle")
    build(synthetic_transactions(n=5_000), params, out, "demo")
    return out


@pytest.fixture(scope="session")
def postgres_url() -> str:
    with PostgresContainer(POSTGRES_IMAGE, driver=None) as pg:
        yield pg.get_connection_url()


@pytest.fixture
def database_url(postgres_url) -> str:
    """The session database, emptied before each test."""
    with psycopg.connect(postgres_url, autocommit=True) as conn:
        for table in TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
    return postgres_url


@pytest.fixture
def make_client(bundle_dir, database_url):
    apps = []

    def factory(**overrides) -> TestClient:
        app = create_app(bundle_dir, database_url, **overrides)
        apps.append(app)
        return TestClient(app)

    yield factory
    for app in apps:
        app.state.store.close()


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
