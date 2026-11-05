# ADR-002: Java decision service, Python inference service

- Status: Accepted
- Date: 2026-11-05
- Decider: project owner

## Context

The project targets backend roles that take ML models to production, including North American banks and fintech companies. Java with Spring Boot appears in bank backend postings (for example, a Scotiabank payments engineering posting reviewed in planning lists Java, Spring Boot, Kafka, and SQL); fintech companies vary, and some use Python or Kotlin. Individual postings do not establish market share.

The owner's public portfolio already shows Python MLOps (AlphaPulse) and Go (Raft-Recovery) but no Java. The model is a LightGBM classifier whose training and preprocessing are naturally in Python.

## Decision

- **Java (Spring Boot) decision service** owns: request validation, idempotency, policy evaluation and versioning, the decision log, model version routing (active, canary, shadow), resilience around model calls, service authentication, and the transactional outbox.
- **Python (FastAPI) inference service** owns: preprocessing shared with training and scoring; it returns a score and the model version and holds no decision logic.
- The services communicate over synchronous HTTP with an OpenAPI contract.

The Java layer must keep these responsibilities. If it degenerates into a pass-through proxy, this decision should be revisited and the layer removed.

## Alternatives considered

- **Python only.** Simplest and sufficient for ML platform roles, but adds little beyond the existing Python portfolio and gives no Java evidence for bank backend roles.
- **Go decision service.** Fits platform teams, but Go is already shown in the portfolio and is less aligned with bank backend postings.
- **Serve the model inside Java (for example, via ONNX).** Removes a network hop, but splits preprocessing across languages and weakens training/serving parity; revisit only if latency measurements require it.

## Consequences

- Two runtimes, two build systems, and a cross-language contract to maintain; contract tests are required.
- Offline/online parity must be tested explicitly (scope S-10).
- Interview narrative: business decisions and reliability in Java, model concerns in Python, with a clear boundary.
- Exact Java, Spring Boot, and Python versions are chosen when each service is scaffolded, after checking current support.
