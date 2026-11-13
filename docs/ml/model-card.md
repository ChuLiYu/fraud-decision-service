# Model card: LightGBM fraud score

Status: current as of code `47642fa` (models, run `m5-client`) and the decision, fusion, and error reports regenerated from those models. Numbers below are copied from the linked generated reports for reading convenience; the reports are authoritative, and this card must be updated when they are regenerated. Structure follows Mitchell et al., "Model Cards for Model Reporting" (2019).

## Model details

| Item | Value |
|---|---|
| Purpose | Score card-not-present transactions for fraud risk, as input to an approve / review / decline policy |
| Type | Gradient-boosted decision trees (LightGBM 4.7.0), binary objective |
| Main settings | 127 leaves, at least 50 rows per leaf, learning rate 0.05, 3,037 trees chosen by early stopping on validation PR-AUC; full list in [`ml/params.yaml`](../../ml/params.yaml) |
| Inputs | 438 numeric and 31 categorical columns after preprocessing: raw transaction and identity fields plus time-safe history features per card and per client key (card, address, and card start day, after the IEEE-CIS 1st-place solution) ([features](../../ml/src/fraud_ml/features.py)) |
| Output | A score in [0, 1]. Not a calibrated probability as is; the expected-cost policy recalibrates it on validation (isotonic regression) |
| Compared with | Logistic regression (baseline) and an MLP; both weaker on every reported metric ([comparison](../../ml/reports/comparison.md)) |
| Version identity | Models manifest params hash `28d4a5d7e8a0` with input file hashes and commit ([manifest](../../ml/src/fraud_ml/manifest.py)) |
| Owner and context | Course project (AI course, primary) and job portfolio ([ADR-004](../decisions/ADR-004-course-first-priority.md)) |

## Intended use

- Ranking transactions for review in an offline, simulated decision system, and comparing decision policies under stated cost assumptions.
- Teaching and demonstrating a leakage-safe ML pipeline and decision-level evaluation.

## Out-of-scope uses

- Real payment decisions or any decision about real people. Data is historical and anonymized, and costs are assumed ([scope](../product/scope.md)).
- Customer or account identification: card and address fields are not a verified identity ([dataset card](../data/dataset-card.md)).
- Redistribution: trained model files stay local because models can memorize rows of Competition Data (dataset card, rules 7B).
- Approving a transaction without a valid score: any failure routes to review ([ADR-003](../decisions/ADR-003-failure-routes-to-review.md)).

## Data

- IEEE-CIS Fraud Detection (Vesta), labelled training tables only: 590,540 transactions, 3.5% fraud, about 182 days. Terms: non-commercial use, no redistribution ([dataset card](../data/dataset-card.md)).
- Split by `TransactionDT` in time order: train 413,378 rows, validation 88,581, test 88,581. Preprocessing is fitted on train only.
- Leakage rules 1–5 in the dataset card each have a test; test labels never influence tuning, fusion, or decision thresholds.

## Metrics and why

- **PR-AUC** as the main ranking metric: with 3.5% fraud, ROC-AUC looks high even for weak models.
- **Precision and recall at fixed review rates**: what a review team with a given capacity would see.
- **Calibration (ECE, Brier)**: needed when scores are multiplied by amounts.
- **Time slices**: drift after the training period.
- **Decision cost per 1,000 transactions** under named cost scenarios: the measure the model is finally judged on.

## Results on the test range

Single seed (42), run `m5-client` ([evaluation](../../ml/reports/runs/m5-client/evaluation.md)):

| Metric | LightGBM | Logistic regression | MLP |
|---|---|---|---|
| PR-AUC | 0.580 | 0.363 | 0.432 |
| ROC-AUC | 0.909 | 0.841 | 0.876 |
| Recall when reviewing 1% (precision) | 0.256 (0.889) | 0.204 (0.711) | 0.218 (0.757) |
| Recall when reviewing 5% (precision) | 0.635 (0.442) | 0.446 (0.310) | 0.506 (0.352) |

LightGBM calibration on test: ECE 0.020, Brier 0.023.

Decision level, base cost scenario, 2% review capacity ([decision report](../../ml/reports/decision/decision.md)):

| Policy | Cost per 1,000 transactions | Saving vs approving everything |
|---|---|---|
| Approve everything | $5,301 | — |
| LightGBM score thresholds | $2,744 | 48% |
| LightGBM expected cost (calibrated probability × amount) | $2,083 | 61% |

The expected-cost policy is the cheapest in all three cost scenarios and four review capacities evaluated.

## Uncertainty

- Seed noise: across three seeds the validation PR-AUC standard deviation was 0.003 for the chosen setting and 0.015 for the previous one ([tuning report](../../ml/reports/tuning/tuning.md)). Gains of about 0.01 PR-AUC between milestones are within that range.
- Validation scores are optimistic: validation drove early stopping and setting selection. Test scores are the unbiased figures.
- Decision-cost differences carry bootstrap 95% intervals in the decision report; all intervals for LightGBM's savings exclude zero.

## Limitations

- **Drift.** PR-AUC falls from 0.81 to 0.56 across the four validation slices after training, and fixed policies send more transactions to review than planned on later test slices (for example up to 2.39% instead of 2% at a 2% capacity) ([figures](../../ml/reports/figures/)). Thresholds and calibration need scheduled refits.
- **Assumed costs.** Review, chargeback, and false-decline costs are scenario parameters, not measurements. Conclusions were checked under three scenarios only.
- **Confounded comparison.** The expected-cost policy differs from score thresholds in two ways (calibration and use of the amount); their separate effects were not measured.
- **Anonymized features.** Most columns are masked, so the model cannot be explained in business terms; no feature meanings are invented.
- **Serving.** History features need past transactions per card; a serving system without an online feature store would need a model trained without them.

## Ethical considerations

- Declines affect legitimate customers. The cost model includes a fixed false-decline cost so that declining small legitimate purchases is not treated as free, and every policy reports its false-decline count.
- No demographic attributes are present, so fairness across groups cannot be assessed; proxies (for example device, email domain, address) may still correlate with protected attributes in real use.
- Results use historical data from one provider and do not transfer to other issuers or regions without re-evaluation.

## Reproduce

From `ml/`, with the dataset downloaded (see [README](../../README.md)): `make all`. Each stage writes a manifest with input hashes, parameters, and commit; reports state hardware, versions, and timing.
