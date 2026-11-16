# Documentation index

Pick the document for your task; do not load everything.

| Task | Authoritative document |
|---|---|
| What the project does and does not do; priorities; phase milestones and acceptance | [product/scope.md](product/scope.md) |
| Why a decision was made | [decisions/](decisions/) (ADRs) |
| Dataset terms, time semantics, splits, leakage rules | [data/dataset-card.md](data/dataset-card.md) |
| Course presentation: slide outline and figure list (draft) | [presentation/outline.md](presentation/outline.md) |
| Model purpose, limits, results summary | [ml/model-card.md](ml/model-card.md) |
| What each pipeline parameter does and how to tune it | [ml/tuning-guide.md](ml/tuning-guide.md) |
| Decision event format (Kafka) | [contracts/events/](contracts/events/) |
| Fault-injection evidence | [operations/reliability-report.md](operations/reliability-report.md) |
| Model results across runs (generated; do not edit) | [../ml/reports/comparison.md](../ml/reports/comparison.md); per-run reports in [../ml/reports/runs/](../ml/reports/runs/); tuning in [../ml/reports/tuning/tuning.md](../ml/reports/tuning/tuning.md) |

## Planned documents (not yet written)

| Document | Owns | Written when |
|---|---|---|
| `contracts/` OpenAPI | Scoring/decision APIs (the MVP serves FastAPI's generated `/openapi.json`) | Before the Java service |
| `docs/domain/decision-policy.md` | Thresholds, review budget, idempotency rules | Phase 2 |
| `docs/operations/model-rollout.md` | Model versions, shadow, canary, rollback rules | Before phase 4 code |

## Proposals (not accepted)

- [Fusion model](proposals/fusion-model.md) — offline ablation accepted (S-18); serving use pending review
- [Jev comparison experiment](proposals/jev-experiment.md) — blocked on real data by dataset terms

## Decision records

- [ADR-001](decisions/ADR-001-new-repository.md) — Start a new repository instead of converting `aml-streaming-lab`
- [ADR-002](decisions/ADR-002-java-decision-python-inference.md) — Java decision service, Python inference service
- [ADR-003](decisions/ADR-003-failure-routes-to-review.md) — Model failure routes to review, never approval
- [ADR-004](decisions/ADR-004-course-first-priority.md) — Course project first, job portfolio second
- [ADR-005](decisions/ADR-005-portfolio-mvp-python-service.md) — Portfolio MVP as one Python decision service ([plan](plans/mvp.md))
- [ADR-006](decisions/ADR-006-outbox-cdc-postgres.md) — Decision events through a Postgres outbox and Debezium ([plan](plans/events.md))

## Language

English documents are authoritative. zh-TW reading versions sit beside their source with a `.zh-TW.md` suffix and never override it:

- [README.zh-TW.md](../README.zh-TW.md)
- [product/scope.zh-TW.md](product/scope.zh-TW.md)
- [proposals/fusion-model.zh-TW.md](proposals/fusion-model.zh-TW.md)
- [proposals/jev-experiment.zh-TW.md](proposals/jev-experiment.zh-TW.md)
- [ml/tuning-guide.zh-TW.md](ml/tuning-guide.zh-TW.md)
