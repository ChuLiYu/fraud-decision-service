# Reliability report

Fault-injection evidence for scope phases 3–4. Each entry states the command, settings, environment, and result. Rerun the command to reproduce; numbers from other machines go in new entries.

## Decision events: publisher down, service killed, consumer replayed (S-6)

Command: `docker compose up --build -d --wait`, then `scripts/events-e2e.sh 200`.

What the script does ([source](../../scripts/events-e2e.sh), [ADR-006](../decisions/ADR-006-outbox-cdc-postgres.md)):

1. Stops Kafka Connect, so committed decisions cannot be published.
2. Commits 200 decisions over HTTP (each returned 200), then sends SIGKILL to the decision service.
3. Starts Connect; compares outbox row ids with the event ids on the topic.
4. Checks the consumer's outcome counts sum to the number of outbox rows.
5. Stops the consumer, rewinds its group to the earliest offset, restarts it, and checks the counts did not change after every message was redelivered.

Result on 2026-11-15 (code from the commit that adds this report; the stack already held 52 earlier events, so totals include them):

| Check | Result |
|---|---|
| Outbox rows after the kill | 252 (52 + 200) |
| Messages on `fraud.decision-events` | 252, 252 distinct event ids |
| Outbox events missing from the topic (lost) | 0 |
| Topic events without an outbox row (phantom) | 0 |
| Outcome counts sum after first delivery | 252 |
| Outcome counts sum after redelivering all 252 messages | 252 |

A run with 50 decisions on a fresh stack gave the same result (52 rows, 0 lost, 0 phantom, recount 0).

Environment: Apple M1, 16 GB RAM, macOS 26.6.2; OrbStack Docker Engine 29.8.2 (VM limit 7.8 GiB); Docker Compose 5.5.0. Images: `postgres:18.7-alpine`, `apache/kafka:4.4.0` (single KRaft node, 3 partitions), `quay.io/debezium/connect:3.7.1.Final`, service image from this repository with the demo model.

Memory of the running stack after the run (`docker stats`): Connect 609 MiB, Kafka 478 MiB, decision service 171 MiB, Postgres 38 MiB, consumer 28 MiB; about 1.3 GB in total.

Limits: one broker, one Connect worker, one consumer; Connect was stopped gracefully, not killed, so Debezium replaying from an older offset after a Connect crash was not exercised here (the consumer's dedupe by `event_id` covers that case and is shown by step 5). No latency measurement: this is a correctness check, not a load test.

## Shadow scoring: correctness after the kill test, and request latency (S-7)

Commands, on a fresh stack (`docker compose down -v`, then `docker compose up --build -d --wait`): `scripts/events-e2e.sh 50`, then `scripts/shadow-latency.sh 300 10`, then `docker compose exec shadow python -m fraud_service.shadow_report` ([plan](../plans/shadow.md)).

Correctness (step 6 of [events-e2e.sh](../../scripts/events-e2e.sh)): after the kill test, and again after rewinding the `shadow-scoring` consumer group and redelivering every message, `shadow_scores` held exactly one row per event: 50 rows, all `scored`.

Latency ([shadow-latency.sh](../../scripts/shadow-latency.sh)): 20 warm-up requests, then 300 sequential requests with the shadow consumer stopped and 300 with it running. Values are the server-side `latency_ms` stored with each decision (scoring time; client-side HTTP time is not included).

| Run | p50 | p99 |
|---|---:|---:|
| Shadow stopped | 5.32 ms | 7.24 ms |
| Shadow running | 5.42 ms | 9.13 ms |
| Difference | +0.10 ms | +1.89 ms |

Tolerance 10 ms (plan A3): pass. An earlier run on the same machine before the final fixes gave +0.20 ms p50 and +1.53 ms p99.

Comparison summary after both scripts (670 events, all scored, `shadow_error` / `invalid_score` / `no_request` 0): outcome agreement 528 / 670 (78.81%); mean |shadow − active score| 0.0577, max 0.2569. Both models are synthetic demo bundles (`--seed` differs), so these numbers only show the comparison works; they say nothing about a real challenger model.

Environment: as in the S-6 entry (Apple M1, 16 GB RAM, macOS 26.6.2; OrbStack Docker Engine 29.8.2, VM limit 7.8 GiB; Docker Compose 5.5.0), code at merge commit d83793c on `claude/shadow-infra`. Memory after the run: Connect 1.0 GiB, Kafka 431 MiB, decision service 157 MiB, shadow consumer 139 MiB, Postgres 41 MiB, outcome consumer 27 MiB.

Limits: one run of each script; requests are sequential, so this does not measure latency under concurrent load or CPU contention from a busy shadow consumer. The shadow consumer shares the host with the decision service, which is the case the check is meant to cover.
