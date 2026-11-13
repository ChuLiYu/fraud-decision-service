# Plan: portfolio MVP

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
| 1 | Serving model | `params.serving.yaml` turns history features off; `make serve-model` writes model, preprocessor, policy, and a manifest to one directory; evaluation report shows the gap to the full model |
| 2 | Demo model | `python -m fraud_ml.demo` generates synthetic rows and writes the same bundle; a test checks the bundle loads and scores |
| 3 | Decision API | `POST /v1/decisions` and `GET /health`; request validation (decimal amount strings); response has outcome, score, model and policy versions, reason; tests for each outcome, invalid input (422), model error and timeout (review) |
| 4 | Idempotency and log | `Idempotency-Key`: same key and body returns the stored decision, different body returns 409; every decision logged; tests |
| 5 | Run and CI | `docker compose up` serves the demo model; `make demo` sends one request per outcome; GitHub Actions runs `ml` and `service` checks and the image build |
| 6 | README | Architecture, one-command run, sample request and response, results with links to reports, limitations |

Out of the MVP: authentication, online history features (S-9), events, Java service, hosted deployment.
