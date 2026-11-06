# Proposal: fusion model (second model combined with LightGBM)

Status: **draft, pending owner review**. Not accepted; not in [scope](../product/scope.md). Acceptance requires a scope update or an ADR.

Reading version (zh-TW): [fusion-model.zh-TW.md](fusion-model.zh-TW.md)

## Idea

Train a second model alongside LightGBM and combine their scores (averaging, weighted blending, or a stacked meta-model) into one fraud score.

## Concerns to resolve before acceptance

1. **Stacking leakage.** A meta-model may be trained only on predictions for a time range later than the base models' training range (for example, validation-range predictions), never on in-sample predictions. This rule would be added to the dataset card's leakage rules.
2. **Partial failure semantics.** If one sub-model succeeds and another times out, the outcome must be defined. The conservative option under [ADR-003](../decisions/ADR-003-failure-routes-to-review.md) is `review` for the whole request; falling back to a single model would need its own ADR.
3. **Versioning.** The model version becomes a composite (fusion version plus each sub-model version); the decision log records all of them.
4. **Evidence threshold.** Fusion must beat LightGBM alone on every time slice by a margin stated in advance, enough to justify added latency, parity checks (S-10), and operating cost.

## Suggested path

Run the second model as a shadow scorer first (S-7). Decide on fusion from the stored active/shadow comparison rather than offline results alone. This adds no new scope until fusion is chosen.

## Open questions for the owner

- Which second model, and why it is expected to make different errors from LightGBM.
- Whether the course grades modelling depth (changes priority; see scope open questions).
