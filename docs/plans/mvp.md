# Plan: portfolio MVP

Status: slices 1–6 implemented; CI not yet run on GitHub. The SQLite store was replaced by PostgreSQL in the [events plan](events.md) ([ADR-006](../decisions/ADR-006-outbox-cdc-postgres.md)).

Implements [ADR-005](../decisions/ADR-005-portfolio-mvp-python-service.md). Each slice ends with a runnable, committed result and its own tests. Fixtures are synthetic only.

```text
client ──HTTP──> service (FastAPI)
                   │ validate request, idempotency (SQLite)
                   │ fraud_ml preprocessing + LightGBM serving model
                   │ threshold policy (versioned file)
                   │ any scoring failure or timeout → review (ADR-003)
                   └ decision log (SQLite)
```

| # | Slice | Acceptance (observable) |
|---|---|---|
| 1 | Serving model | `fraud_ml.bundle` turns history features off; `make serve-model` writes model, preprocessor, policy, and `bundle.json` to one directory, and an aggregate report to `ml/reports/serving/` |
| 2 | Demo model | `make demo-model` (`fraud_ml.bundle --demo`) generates synthetic rows and writes the same bundle; a test checks the bundle loads and scores |
| 3 | Decision API | `POST /v1/decisions` and `GET /health`; request validation (decimal amount strings); response has outcome, score, model and policy versions, reason; tests for each outcome, invalid input (422), model error and timeout (review) |
| 4 | Idempotency and log | `Idempotency-Key`: same key and body returns the stored decision, different body returns 409; every decision logged; tests |
| 5 | Run and CI | `docker compose up` serves the demo model; `scripts/demo.sh` sends sample requests; GitHub Actions runs `ml` and `service` checks and the image build |
| 6 | README | Architecture, one-command run, sample request and response, results with links to reports, limitations |

Out of the MVP: authentication, online history features (S-9), events, Java service, hosted deployment.
