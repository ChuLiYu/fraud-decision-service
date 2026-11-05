# ADR-001: Start a new repository instead of converting aml-streaming-lab

- Status: Accepted
- Date: 2026-11-05
- Decider: project owner

## Context

`aml-streaming-lab` (GitHub `ChuLiYu/aml-streaming-lab`, local directory `kafka`) is an archived Go prototype: a synthetic transaction producer, Redpanda, a Redis sliding-window structuring rule, and Prometheus/Grafana. It has one commit. Its README records that the code was never compiled, run, or benchmarked in the 2026-10-29 review, and lists unresolved gaps: offset commit semantics, replay and deduplication, per-account ordering under concurrent workers, non-atomic Redis decisions, and floating-point money.

This project differs in domain (fraud scoring rather than AML structuring), data (IEEE-CIS rather than synthetic), languages (Java and Python rather than Go), and purpose (serving an ML model rather than a single heuristic rule).

## Decision

Create a new repository, `fraud-decision-service`. Keep `aml-streaming-lab` archived as a record of the earlier prototype. Copy monitoring configuration from it only where useful.

## Alternatives considered

- **Convert `aml-streaming-lab` in place.** Rejected: the repository name and history would describe a different system, and little code would survive the change of domain and language.
- **Rename `aml-streaming-lab`.** Rejected: keeps an unrelated single-commit history and loses the archived prototype as a separate reference.

## Consequences

- The new repository starts with documentation that matches its actual scope.
- The predecessor's unresolved gaps become explicit acceptance criteria here (see [scope](../product/scope.md), phase 3), which gives a coherent "lessons applied" narrative.
- Archiving `aml-streaming-lab` on GitHub, if wanted, is a separate manual action by the owner.
