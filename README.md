# Fraud Decision Service

A fraud decision service that takes an ML model to production: versioned scoring, policy-based approve/review/decline decisions, an auditable decision log, and safe model rollout.

Reading version (zh-TW): [README.zh-TW.md](README.zh-TW.md)

> **Status:** phase 1 (model and evaluation, the AI course project) is runnable; the services in the architecture below (phases 2–5) are planned and start after the course submission ([ADR-004](docs/decisions/ADR-004-course-first-priority.md)).

## Why this project

Training a fraud model is one part of the problem. This project focuses on the system between a trained model and the decision a customer experiences:

- Model scores and business policy are versioned separately.
- Every decision can be traced and replayed.
- A failing or slow model never silently approves a transaction.
- New model versions are evaluated in shadow and canary modes before promotion, and can be rolled back.

## Run phase 1

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
| `make all` | Everything above except `tune`, from download to figures | all of the above |

Every stage refuses inputs whose manifest does not match the current `ml/params.yaml`. What each parameter does: [tuning guide](docs/ml/tuning-guide.md).

## Planned architecture (phases 2–5)

```text
Transaction request
  -> Java decision service (Spring Boot)
       -> sync HTTP -> Python inference service (active / canary model)
       -> Redis online features
       -> PostgreSQL: decision log + outbox (one transaction)
            -> Debezium CDC -> Kafka `decisions`
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
