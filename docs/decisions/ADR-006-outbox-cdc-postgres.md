# ADR-006: Decision events through a Postgres outbox and Debezium

- Status: Accepted
- Date: 2026-11-15
- Decider: project owner (requested reaching the Kafka stage next)
- Changes: replaces SQLite in the MVP service ([ADR-005](ADR-005-portfolio-mvp-python-service.md)) with PostgreSQL; orders phase 3 S-6 ahead of the remaining phase 2 items (S-5 authentication, S-10 parity, Java service)

## Context

Scope S-6 requires decision events published through a transactional outbox and CDC to Kafka. Phase 3 acceptance requires that killing the decision service between database commit and event publication loses no event and creates no phantom event, and that consumers tolerate duplicate delivery and restart.

The MVP stores decisions in SQLite inside the service container. Debezium reads PostgreSQL's write-ahead log through logical decoding; a SQLite file shared between containers is not a sound CDC source.

## Decision

- **Store:** PostgreSQL 18 replaces SQLite. One transaction inserts the decision row and its outbox row; an idempotent replay or conflict inserts neither.
- **Relay:** Debezium PostgreSQL connector (`pgoutput`) on Kafka Connect with the Outbox Event Router reads the `outbox` table and publishes the payload to the `fraud.decision-events` topic, keyed by `decision_id`. The service never talks to Kafka.
- **Contract:** the event payload is the JSON Schema [`decision-made.v1`](../contracts/events/decision-made.v1.schema.json); tests validate every emitted payload against it. A schema registry and its compatibility check are S-13, later.
- **Consumer:** a Python consumer (`fraud_service.consumer`) projects events into per-model outcome counts. It records each `event_id` and updates the counts in one database transaction, then commits the Kafka offset, so redelivery after a crash is a no-op. Offsets are committed only after processing (at-least-once delivery, effectively-once effect).
- **Scope of the demo:** consumer tables live in the same database as the decision log, for one-command setup. In production they would be separate.
- **Outbox rows are kept** (not deleted after insert) so the kill test can compare the outbox with the topic. Retention is out of scope for the demo.

## Alternatives considered

- **SQLite outbox with a polling relay.** Lighter (no Postgres, no Connect), but it is not CDC, so it would need a scope change; polling also adds a relay to operate and test.
- **Service publishes to Kafka after commit.** Loses events when the process dies between commit and publish; this is the failure the outbox exists to prevent.
- **Debezium Server instead of Kafka Connect.** Fewer moving parts, but Kafka Connect is the more common deployment and gives a REST API for connector status.

## Consequences

- `docker compose up` now starts five containers (Postgres, Kafka, Kafka Connect, decision service, consumer) and needs more memory: about 1.3 GB measured on the owner's machine ([reliability report](../operations/reliability-report.md)).
- Service tests need Docker for a Postgres container (Testcontainers); CI already provides Docker.
- The service has no direct Kafka dependency, so event publication cannot affect decision latency or outcome.
