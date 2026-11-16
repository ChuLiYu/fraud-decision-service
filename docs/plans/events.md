# Plan: decision events (phase 3, S-6)

Status: slices 7–10 implemented; kill test passed locally ([reliability report](../operations/reliability-report.md)) and in GitHub CI (run 43806359049, commit 18fb056).

Implements [ADR-006](../decisions/ADR-006-outbox-cdc-postgres.md). Continues the numbering of the [MVP plan](mvp.md).

```text
client ─HTTP─> decision service ──one transaction──> Postgres: decisions + outbox
                                                        │ WAL (pgoutput)
                                     Debezium outbox router (Kafka Connect)
                                                        │
                                        Kafka topic fraud.decision-events (key decision_id)
                                                        │
                                consumer: dedupe by event_id + outcome counts (Postgres)
```

| # | Slice | Acceptance (observable) |
|---|---|---|
| 7 | Contract | [`decision-made.v1`](../contracts/events/decision-made.v1.schema.json) JSON Schema; tests validate emitted payloads against it |
| 8 | Postgres store and outbox | Decision and outbox row written in one transaction; replay and 409 write no outbox row; concurrent duplicates write one of each; failure-path decisions (review) emit events too; tests on a Postgres container |
| 9 | Consumer | Processing the same event twice changes counts once; offset committed only after the database commit; tests |
| 10 | Compose and kill test | `docker compose up` runs Postgres, Kafka, Connect (connector registered automatically), service, consumer; `scripts/events-e2e.sh` stops Connect, commits decisions, kills the service, restarts Connect and the consumer, and checks that every outbox row reaches the topic once and is counted once; CI runs it |

Out of this plan: shadow scoring (S-7), schema registry (S-13), authentication (S-5), parity checks (S-10).
