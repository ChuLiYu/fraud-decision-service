# Fraud Decision Service

A fraud decision service that takes an ML model to production: versioned scoring, policy-based approve/review/decline decisions, an auditable decision log, and safe model rollout.

Reading version (zh-TW): [README.zh-TW.md](README.zh-TW.md)

> **Status: planning.** No runnable code yet. This README describes intended scope, not delivered features. Results appear below only after they are measured.

## Why this project

Training a fraud model is one part of the problem. This project focuses on the system between a trained model and the decision a customer experiences:

- Model scores and business policy are versioned separately.
- Every decision can be traced and replayed.
- A failing or slow model never silently approves a transaction.
- New model versions are evaluated in shadow and canary modes before promotion, and can be rolled back.

## Planned architecture

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

## Results

None yet. Each result will link to its report with hardware, data volume, and settings.

## Documentation

Start at [docs/INDEX.md](docs/INDEX.md).
