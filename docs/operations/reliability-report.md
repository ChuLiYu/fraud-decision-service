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
