# ADR-003: Model failure routes to review, never approval

- Status: Accepted
- Date: 2026-11-05
- Decider: project owner

## Context

The decision service depends on the inference service and on Redis online features. Either can be unavailable, slow, or return errors. A fallback has to be chosen for each decision request when a score cannot be obtained in time.

## Decision

- If a valid score from the active model is not available within the configured timeout (after bounded retries, or immediately when the circuit is open), the decision is `review`.
- The decision record marks the degraded path and its cause (timeout, circuit open, error, invalid response).
- Missing online features follow a documented degraded mode defined with the feature contract; a request must not be scored as if missing features were known low-risk values.
- Shadow and canary failures never affect the active decision; a canary failure falls back to the active model before falling back to `review`.
- The system never returns `approve` without a valid score from a known model version.

## Alternatives considered

- **Fail open (approve).** Rejected: converts an availability incident into undetected fraud loss.
- **Fail closed (decline).** Rejected as the default: blocks legitimate customers during outages. May be a policy option for specific high-risk segments later.
- **Return an error to the caller.** Rejected as the default: pushes the decision to every client inconsistently; `review` keeps the outcome explicit and recorded.

## Consequences

- Review volume rises during incidents; dashboards must show degraded-decision rate so this is visible.
- Fault-injection tests are required for each failure mode (scope phase 2 acceptance).
- Review capacity is a scenario parameter in this project, not a measured operational figure.
