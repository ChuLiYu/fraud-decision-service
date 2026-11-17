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

Commands, on a fresh stack (`docker compose down -v`, then `docker compose up --build -d --wait`), in the CI order: `scripts/events-e2e.sh 50`, then `scripts/shadow-latency.sh 2 100 none` (the CI step), then `scripts/shadow-latency.sh 5 200 10` (the A3 check) ([plan](../plans/shadow.md)).

Correctness (step 6 of [events-e2e.sh](../../scripts/events-e2e.sh)): after the kill test, and again after rewinding the `shadow-scoring` consumer group and redelivering every message, `shadow_scores` held exactly one row per event: 50 rows, all `scored`. After all three commands the stack held 2,550 events and 2,550 shadow rows.

Latency ([shadow-latency.sh](../../scripts/shadow-latency.sh)): 50 warm-up requests, then 5 pairs of 200-request sequential blocks, one block with the shadow consumer stopped and one with it running; the order alternates per pair (off/on, on/off, ...) to cancel drift in host load. Before each running block the shadow backlog is cleared; during it the consumer must score events (it scored 170–190 of 200 while each block was sending). Values are the server-side `latency_ms` stored with each decision (scoring time; client-side HTTP time is not included).

| Pair | Order | Off p50 / p99 (ms) | On p50 / p99 (ms) | Δ p50 | Δ p99 |
|---|---|---|---|---:|---:|
| 1 | off, on | 5.34 / 8.17 | 5.95 / 17.61 | +0.61 | +9.44 |
| 2 | on, off | 5.27 / 6.66 | 5.43 / 8.46 | +0.16 | +1.80 |
| 3 | off, on | 5.41 / 8.13 | 7.04 / 16.71 | +1.63 | +8.58 |
| 4 | on, off | 6.36 / 12.57 | 6.28 / 14.62 | −0.08 | +2.05 |
| 5 | off, on | 6.08 / 7.91 | 7.54 / 46.14 | +1.46 | +38.23 |
| **Median** | | | | **+0.61** | **+8.58** |

Tolerance 10 ms on the median (plan A3): pass, with a small margin at p99. The shadow consumer runs on the same host as the decision service, and while it is scoring the p99 tail rises in most pairs (up to +38 ms in one); p50 barely moves. This is CPU contention on one machine, not the request path waiting on shadow work, and is the main reason to run the shadow consumer on separate capacity in a real deployment. The host was not idle: load average 4.9 at the start of the run and 16.2 at the end, from other applications.

CI step (2 pairs of 100, not judged): median Δ p50 +0.76 ms, Δ p99 +0.16 ms; shadow scored 90 and 77 of 100 during the running blocks.

Superseded measurement: an earlier version of the script ran one 300-request block each way after a fixed 5-second pause, without checking that the shadow consumer was scoring during the running block; its results (+0.10 / +1.89 ms, and on a busier host a −8.59 / +21.48 ms fail) are not evidence for A3.

Comparison summary: on an earlier fresh-stack run (670 events, all scored, `shadow_error` / `invalid_score` / `no_request` 0), outcome agreement was 528 / 670 (78.81%); mean |shadow − active score| 0.0577, max 0.2569. Both models are synthetic demo bundles (`--seed` differs), so these numbers only show the comparison works; they say nothing about a real challenger model.

Environment: as in the S-6 entry (Apple M1, 16 GB RAM, macOS 26.6.2; OrbStack Docker Engine 29.8.2, VM limit 7.8 GiB; Docker Compose 5.5.0), code on `claude/shadow-infra` at the commit that adds this entry.

Limits: one run of the A3 check; requests are sequential, so this does not measure latency under concurrent load.
