# Proposal: Jev (TypeSafe AI) as a comparison experiment

Status: **draft, pending owner review**. Not accepted; not in [scope](../product/scope.md). Acceptance requires a scope update or an ADR.

Reading version (zh-TW): [jev-experiment.zh-TW.md](jev-experiment.zh-TW.md)

## What Jev is (as of 2026-11-05)

A "System One" model from TypeSafe AI, released in early access in September 2026. It returns no text: each call answers typed questions with a value from options defined in advance, plus a probability. Target tasks are classification, routing, scoring, and extraction. It is available only as a hosted API (for example, Cloudflare Workers AI and DigitalOcean Serverless Inference); weights and architecture are not published.

Most performance, latency (70–500 ms end-to-end), and pricing figures are vendor-reported. No independent benchmark on tabular or fraud data was found.

Sources:

- https://www.digitalocean.com/resources/articles/what-is-jev
- https://www.datacamp.com/blog/system-one-models-jev
- https://www.eesel.ai/blog/typesafe-jev
- https://www.vktr.com/ai-platforms/chatgpt-cocreator-launches-typesafe-ai-with-jev/

## Assessment

Not suitable as the active model or as part of the serving path.

| Issue | Conflict with this project |
|---|---|
| Rows are sent to a third-party API | Dataset terms of use are unverified; sending rows externally may be prohibited |
| No tabular or fraud benchmarks | IEEE-CIS has hundreds of anonymized numeric columns, where gradient-boosted trees are strong |
| Vendor-reported latency of 70–500 ms | Consumes much of the synchronous decision latency budget (phase 2) |
| Closed, hosted-only, can change without notice | Conflicts with reproducibility (S-1) and offline/online parity (S-10) |
| Returns a probability without a reason | Decisions must be explainable and replayable |
| Scope excludes LLM features added for demonstration | Jev is close in nature even though it does not generate text |

## Possible experiment, if accepted

An offline comparison only, outside the serving path:

1. Preconditions: Kaggle terms permit sending rows to a third party; scope or ADR updated.
2. Sample from the time-based test range; send raw fields only, no engineered features from future data.
3. Compare with LightGBM on PR-AUC, precision/recall at fixed review rates, calibration, cost per decision, and measured latency, with hardware and settings recorded.

## Open questions for the owner

- Whether the Kaggle terms allow sending data to an external model API.
- Whether the course values a comparison with a hosted foundation model.
