# Scope

Status: accepted by the owner on 2026-11-05. Changes to scope require updating this file; significant trade-offs also get an ADR.

Reading version (zh-TW): [scope.zh-TW.md](scope.zh-TW.md)

## Problem

Given a card-not-present transaction, produce a fraud risk score, turn it into an approve/review/decline decision under a versioned policy, and keep a complete record so any decision can be explained, replayed, and evaluated once labels arrive. All decisions are simulated.

## Positioning

A portfolio project for backend roles specializing in systems that take ML models to production, with secondary fit for ML platform/MLOps roles. Model training is kept sound but minimal; engineering effort goes to serving, decision reliability, rollout, and observability.

## Readers

- Hiring managers and interviewers reviewing the repository.
- The owner and contributing agents implementing it.

## In scope

| ID | Capability |
|---|---|
| S-1 | Reproducible training and time-based evaluation of a baseline and a LightGBM model |
| S-2 | Python inference service returning score and model version, sharing preprocessing with training |
| S-3 | Java decision service: validation, idempotency, versioned policy, decision log |
| S-4 | Failure handling: timeouts, circuit breaking, bounded retries; failures route to review ([ADR-003](../decisions/ADR-003-failure-routes-to-review.md)) |
| S-5 | Service-to-service authentication (OAuth2 client credentials) |
| S-6 | Decision events published through a transactional outbox and CDC to Kafka |
| S-7 | Asynchronous shadow scoring and active/shadow comparison |
| S-8 | Model version registry, canary routing, and rollback |
| S-9 | Request-time online features in Redis with a defined degraded mode |
| S-10 | Offline/online score parity checks |
| S-11 | Observability: traces across services, latency/error/lag metrics, score and missing-value distributions |
| S-12 | Historical replay and load testing with recorded reports |
| S-13 | Versioned event schemas with a demonstrated compatible change |
| S-14 | Local Kubernetes deployment with Helm |
| S-16 | Delayed-label join and post-hoc quality metrics (optional) |

## Out of scope

- Real payments, card networks, banks, or customer data.
- AML, KYC, or regulatory reporting.
- Automatic retraining or automatic promotion of models.
- A self-built feature store, service mesh, or multi-region deployment.
- LLM features added for demonstration.
- Cloud deployment and infrastructure as code (AWS, Terraform); the owner demonstrates these through other experience.
- Account or identity linking not supported by the data (card fields are not treated as verified customer identity).

## Architecture summary

Decisions are synchronous: the caller waits for approve/review/decline. Kafka carries events whose processing must not add request latency (shadow scoring, label joins, replay). Service split is recorded in [ADR-002](../decisions/ADR-002-java-decision-python-inference.md).

## Phases and acceptance

A phase is done only when every criterion has evidence in the repository (test, report, or CI run). Targets are not results.

### Phase 1 — Trustworthy model (S-1)

- One command reproduces data preparation, training, and evaluation from the downloaded dataset.
- Train/validation/test split by `TransactionDT`; preprocessing fitted on training data only.
- Report includes ROC-AUC, PR-AUC, precision/recall at fixed review rates, a calibration check, and results by time slice, compared with a baseline.
- Dataset limitations documented in the [dataset card](../data/dataset-card.md).

### Phase 2 — Synchronous decision service (S-2 to S-5, S-10)

- OpenAPI contracts for scoring and decision APIs exist before implementation.
- A decision stores request hash, model version, policy version, score, outcome, and latency.
- Concurrent duplicate requests with one idempotency key produce exactly one decision.
- With the inference service down, slow, or erroring, decisions are `review` and record the degraded path.
- Unauthenticated service calls are rejected.
- Offline and online scores match within a stated tolerance on a fixed sample.
- CI runs unit tests and Testcontainers integration tests.

### Phase 3 — Events and shadow scoring (S-6, S-7, S-13)

- Killing the decision service between database commit and event publication loses no event and creates no phantom event.
- Consumers tolerate duplicate delivery and restart without double-processing.
- Shadow model scores do not change request latency beyond a stated tolerance; comparison results are stored.
- A backward-compatible event schema change passes the registry compatibility check in CI.

### Phase 4 — Rollout, features, and observability (S-8, S-9, S-11, S-12, S-16)

- Canary routing sends a configured share of traffic to a candidate model; rollback restores the previous version in one step.
- With Redis unavailable, the service follows the documented degraded mode.
- A trace shows each hop of one decision; dashboards cover latency, errors, consumer lag, missing-feature rate, and score distribution.
- A load-test report states hardware, data volume, concurrency, throughput, p99 latency, and error rate.

### Phase 5 — Kubernetes deployment (S-14)

- The stack deploys to a local Kubernetes cluster with Helm; new model versions receive traffic only after readiness.

## Constraints

- Raw data, model binaries, and credentials are never committed.
- Monetary amounts use decimal or integer minor units, not binary floating point.
- Tool and library versions are pinned when each component is scaffolded, after checking current support.

## Open questions

| Question | Blocks |
|---|---|
| Kaggle competition terms for data use and publishing derived artifacts (not yet verified) | Publishing models, sample data, or demos |
| AI course rules on publishing the project and individual vs team contribution | Making the repository public; resume wording |
| Course deadline and required deliverables | Phase 1 ordering and format |
