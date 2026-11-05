# Documentation index

Pick the document for your task; do not load everything.

| Task | Authoritative document |
|---|---|
| What the project does and does not do; phase acceptance | [product/scope.md](product/scope.md) |
| Why a decision was made | [decisions/](decisions/) (ADRs) |
| Dataset terms, time semantics, splits, leakage rules | [data/dataset-card.md](data/dataset-card.md) |

## Planned documents (not yet written)

| Document | Owns | Written when |
|---|---|---|
| `contracts/` (OpenAPI, event schemas) | Scoring/decision APIs, decision record and event formats | Before phase 2 code |
| `docs/domain/decision-policy.md` | Thresholds, review budget, idempotency rules | Phase 2 |
| `docs/operations/model-rollout.md` | Model versions, shadow, canary, rollback rules | Before phase 4 code |
| `docs/ml/model-card.md` + evaluation report | Metrics, calibration, slices, limits | End of phase 1 |
| `docs/operations/reliability-report.md` | Fault-injection and load-test evidence | Phases 3–4 |

## Decision records

- [ADR-001](decisions/ADR-001-new-repository.md) — Start a new repository instead of converting `aml-streaming-lab`
- [ADR-002](decisions/ADR-002-java-decision-python-inference.md) — Java decision service, Python inference service
- [ADR-003](decisions/ADR-003-failure-routes-to-review.md) — Model failure routes to review, never approval

## Language

English documents are authoritative. zh-TW reading versions sit beside their source with a `.zh-TW.md` suffix and never override it:

- [README.zh-TW.md](../README.zh-TW.md)
- [product/scope.zh-TW.md](product/scope.zh-TW.md)
