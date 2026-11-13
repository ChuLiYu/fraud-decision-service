# ADR-004: Course project first, job portfolio second

- Status: Accepted; timing of later phases changed by [ADR-005](ADR-005-portfolio-mvp-python-service.md)
- Date: 2026-11-05
- Decider: project owner

## Context

The original [scope](../product/scope.md) positioned the project as a backend portfolio with model training kept minimal. The owner has since set the AI course project as the primary goal and the job portfolio as secondary.

The course rubric, provided by the owner, weights four criteria at 25% each:

- **Creativity:** an original idea or approach; standard methods with minor twists score lower.
- **Presentation:** organized, engaging, clear on technical details and results.
- **Comprehensiveness:** the full ML process: data preparation, model development, evaluation, and discussion; depth in feature engineering and hyperparameter tuning is named explicitly.
- **Coding implementation:** efficient, well-structured, thoroughly commented code that runs without errors.

Against this rubric, the earlier plan lacked feature engineering, tuning, presentation materials, and a creative angle beyond a standard LightGBM model.

## Decision

- Phase 1 becomes the course project and is expanded: a staged pipeline with versioned preprocessing artifacts, time-safe feature engineering, tuning on validation, a comparison model with a fusion ablation, decision-level evaluation under review-rate and cost scenarios, and presentation materials (scope S-1, S-17 to S-19).
- The creative angle is evaluating the model as a decision system (review budget, costs, time drift) rather than as a classifier alone. It also links the course work to the later serving phases.
- Phases 2 to 5 (serving, events, rollout, deployment) start after the course submission.
- Work proceeds in small milestones (M0 to M4), each ending with a runnable, committed result.
- Code comments follow the rubric: modules and functions carry docstrings explaining purpose and ML reasoning.

## Alternatives considered

- **Keep the portfolio-first positioning.** Rejected: scores low on comprehensiveness and creativity, which make up half of the course grade.
- **Do both in parallel.** Rejected: splits effort before the course deadline, which is not yet known.
- **Make the Jev comparison the main creative angle.** Not chosen: it depends on whether dataset rows may be sent to an external API, which is unresolved. It stays optional ([proposal](../proposals/jev-experiment.md)).

## Consequences

- The Java decision service and later phases are delayed; the portfolio is complete later.
- Phase 1 carries more ML work and more leakage risk (feature engineering, stacking), so the leakage rules in the [dataset card](../data/dataset-card.md) need tests.
- If the course deadline is short, the cut order in the scope applies: drop the Jev experiment and fusion first.

## Update 2026-11-05

The dataset terms were verified: rules section 7B prohibit transmitting the data to third parties, so the Jev comparison cannot use the competition data and is no longer part of the cut order ([dataset card](../data/dataset-card.md)).
