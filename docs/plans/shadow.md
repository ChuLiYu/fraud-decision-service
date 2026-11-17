# Plan: shadow scoring (phase 3, S-7)

Status: slices 11–13 implemented.

Continues the [events plan](events.md). Shadow scoring runs off the request path: a consumer of the decision events scores each event's request with a second model and stores the comparison. Failures are recorded and never affect active decisions ([ADR-003](../decisions/ADR-003-failure-routes-to-review.md)).

```text
decision service ─outbox─> Kafka fraud.decision-events (now with the scored request)
                                 ├ consumer: outcome counts (S-6)
                                 └ shadow consumer: shadow bundle scores request.fields
                                        → shadow_scores (event_id primary key)
                                        → python -m fraud_service.shadow_report
```

| # | Slice | Author | Acceptance (observable) |
|---|---|---|---|
| 11 | Event carries the request | Codex | `decision.made` v1 gains an optional `request` object (compatible addition; old payloads still validate); non-finite numbers are rejected with 422 so the event can always be stored |
| 12 | Shadow consumer and summary | Codex | One row per `event_id` with active and shadow model versions, scores, and outcomes; shadow errors, invalid scores, and events without `request` are stored with their status; duplicates skipped; summary reports status counts, agreement, confusion matrix, and score differences; `fraud_ml.bundle --seed` builds a second demo model |
| 13 | Stack and evidence | Claude | Image builds the shadow bundle; compose runs the shadow consumer; `scripts/events-e2e.sh` checks one comparison per event after the kill test and after rewinding the shadow group; `scripts/shadow-latency.sh` alternates blocks with the shadow consumer stopped and running (it must be scoring during each running block) and shows the median p50 and p99 difference over the pairs is at most 10 ms ([reliability report](../operations/reliability-report.md)); CI runs both, with the latency script in its no-threshold mode because shared runners are too noisy to gate on latency |

Out of this plan: schema registry (S-13), canary routing (S-8), a real shadow model trained on the dataset (the demo stack uses two synthetic demo models).
