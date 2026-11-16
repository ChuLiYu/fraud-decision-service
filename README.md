# Fraud Decision Service

A fraud decision service that takes an ML model to production: versioned scoring, policy-based approve/review/decline decisions, an auditable decision log, and safe model rollout.

Reading version (zh-TW): [README.zh-TW.md](README.zh-TW.md)

> **Status:** runnable MVP. One Python decision service serves a LightGBM model behind an approve / review / decline API ([ADR-005](docs/decisions/ADR-005-portfolio-mvp-python-service.md)) and publishes every decision to Kafka through a transactional outbox and Debezium ([ADR-006](docs/decisions/ADR-006-outbox-cdc-postgres.md)); the model work behind it (phase 1) is complete with reports. Shadow scoring, authentication, and the Java decision service are later phases.

## Quick start (no dataset needed)

Requirements: Docker with Compose, about 1.5 GB of free memory for the containers.

```bash
docker compose up --build -d --wait
scripts/demo.sh
scripts/events-e2e.sh   # kill test for decision events (about a minute)
```

The image trains a small **demo model on synthetic rows** at build time, because the IEEE-CIS data and models trained on it may not be redistributed. `/health` reports `"model_kind": "demo"`. To serve the real model, download the data (below), run `make -C ml serve-model`, then `docker compose -f compose.yaml -f compose.real.yaml up`.

Example:

```bash
curl -s localhost:8000/v1/decisions -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: order-42' \
  -d '{"transaction_id": 42, "timestamp_seconds": 90000, "amount": "950.00",
       "fields": {"ProductCD": "C", "DeviceType": "mobile", "D1": 1}}'
```

```json
{"decision_id": "…", "transaction_id": 42, "outcome": "decline",
 "reason": "score_at_or_above_decline_threshold", "score": 0.90,
 "model_version": "demo-…", "policy_version": "thresholds-base-review0.02-…"}
```

## What the service does

```text
client ─HTTP─> FastAPI decision service
                 ├ validate: decimal-string amount, known raw fields only (422 otherwise)
                 ├ idempotency: same key + same body → stored decision; different body → 409
                 ├ score: fraud_ml features + preprocessing (same code as training) → LightGBM
                 ├ policy: versioned score thresholds fitted on validation (decision evaluation)
                 ├ failure rule: model error, timeout, or invalid score → review, never approve
                 └ one Postgres transaction: decision log row + outbox row
                      (request hash, model and policy versions, outcome, latency)
Postgres WAL ─> Debezium outbox router (Kafka Connect) ─> Kafka `fraud.decision-events`
                                                              └ consumer: outcome counts per model,
                                                                deduplicated by event_id
```

- Failure handling follows [ADR-003](docs/decisions/ADR-003-failure-routes-to-review.md) and is tested for errors, timeouts, and NaN or out-of-range scores.
- Concurrent requests with one `Idempotency-Key` store exactly one decision and one event (database constraint, tested).
- Events follow a versioned JSON Schema ([decision-made v1](docs/contracts/events/decision-made.v1.schema.json)); the service never calls Kafka, so publishing cannot slow or change a decision. With Kafka Connect stopped and the service killed after 200 commits, all events arrived once restarted, none lost or invented, and replaying the whole topic to the consumer changed no counts ([reliability report](docs/operations/reliability-report.md)).
- The served model has no per-card history features (no online feature store yet); on the test range it scores PR-AUC 0.583 against 0.580 for the full model, single seed ([serving report](ml/reports/serving/serving.md), [evaluation](ml/reports/runs/m5-client/evaluation.md)).
- CI ([workflow](.github/workflows/ci.yml)) runs lint and tests for `ml/` and `service/` (with a Postgres container), builds the stack, and runs the demo and kill-test scripts.

| Path | Content |
|---|---|
| `ml/` | Pipeline: data checks, time split, features, training, evaluation, decision analysis, serving bundle |
| `service/` | FastAPI decision service, Kafka consumer, and their tests |
| `deploy/connect/` | Debezium outbox connector configuration and registration |
| `docs/` | Scope, decisions (ADRs), dataset card, model card, plans |

## Run the model pipeline (phase 1)

Requirements: [uv](https://docs.astral.sh/uv/) (it installs the pinned Python 3.14 and libraries from `ml/uv.lock`), about 16 GB RAM, and the IEEE-CIS data. The data is not in this repository: accept the competition rules on Kaggle, then either run `make fetch` with your own Kaggle API token or place `train_transaction.csv` and `train_identity.csv` in `data/raw/`.

From `ml/`:

| Command | Stage | Writes |
|---|---|---|
| `make setup` | Create the environment from the lock file | `ml/.venv` |
| `make check` | Lint and unit tests (synthetic data only) | — |
| `make prepare` | Validate, add time-safe features, split by time, fit preprocessing on train | `data/processed/` + manifest |
| `make train` | Logistic regression, LightGBM, MLP | `data/models/` + manifest (local only) |
| `make evaluate RUN=<name>` | Validation and test metrics | `ml/reports/runs/<name>/` |
| `make compare` | Table of all runs | `ml/reports/comparison.md` |
| `make tune` | Validation-only grid and seed-noise study | `ml/reports/tuning/` |
| `make fuse` | Offline fusion ablation | `ml/reports/fusion/` |
| `make decide` | Approve / review / decline policies and their cost | `ml/reports/decision/` |
| `make errors` | Error analysis by segment | `ml/reports/errors/` |
| `make figures` | Presentation figures from the reports | `ml/reports/figures/` |
| `make serve-model` | Serving bundle for the service (history features off) | `data/serving/` (local only) + `ml/reports/serving/` |
| `make all` | Everything above except `tune`, from download to figures | all of the above |

Every stage refuses inputs whose manifest does not match the current `ml/params.yaml`. What each parameter does: [tuning guide](docs/ml/tuning-guide.md).

## Planned architecture (later phases)

```text
Transaction request
  -> Java decision service (Spring Boot; the MVP uses Python, ADR-005)
       -> sync HTTP -> Python inference service (active / canary model)
       -> Redis online features
       -> PostgreSQL: decision log + outbox (one transaction)      [done in the MVP]
            -> Debezium CDC -> Kafka `fraud.decision-events`       [done in the MVP]
                 -> shadow-scoring consumer -> inference service (shadow model)
Offline training and evaluation (Python, LightGBM) -> MLflow model registry
OpenTelemetry -> Prometheus / Grafana
```

Decisions are synchronous; Kafka carries events that must not add request latency. See [scope](docs/product/scope.md) for phases and acceptance criteria.

## Data

Uses the IEEE-CIS Fraud Detection dataset from Kaggle, replayed as simulated traffic. Raw data is not stored in this repository. See the [dataset card](docs/data/dataset-card.md) for terms, time semantics, and limitations.

## Boundaries

- Simulated decisions only. No real payments, banks, or customer data.
- Fraud scoring, not AML or KYC compliance.
- Performance figures, when published, are lab measurements on stated hardware.

## Results (phase 1)

Test range, 88,581 transactions after the training and validation periods; hardware, versions, and settings are in each report.

- LightGBM test PR-AUC 0.580, against 0.363 for logistic regression and 0.432 for an MLP ([comparison](ml/reports/comparison.md)).
- Turning scores into decisions: a policy using calibrated probability × amount costs $2,083 per 1,000 transactions at a 2% review capacity, against $5,301 for approving everything and $2,744 for LightGBM score thresholds, under assumed costs; it is cheapest in all three cost scenarios tested ([decision report](ml/reports/decision/decision.md)).
- Negative results are reported too: fusing LightGBM with the MLP did not pass the rule fixed before the run ([fusion report](ml/reports/fusion/fusion.md)).
- Limits, drift, and intended use: [model card](docs/ml/model-card.md).

## Documentation

Start at [docs/INDEX.md](docs/INDEX.md).
