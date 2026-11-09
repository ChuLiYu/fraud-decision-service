# Dataset card: IEEE-CIS Fraud Detection

Status: draft. Terms of use verified on 2026-11-05; structure figures measured on 2026-11-08.

## Source

- Kaggle competition "IEEE-CIS Fraud Detection": https://www.kaggle.com/c/ieee-fraud-detection
- Access requires a Kaggle account that has accepted the competition rules. Data is downloaded by script into an ignored `data/` directory and never committed.

## Terms of use (verified 2026-11-05)

Source: competition rules, section 7 "Competition Data", read at https://www.kaggle.com/competitions/ieee-fraud-detection/rules on 2026-11-05. Paraphrased; the rules page is authoritative.

- **Permitted use (7A):** non-commercial only, explicitly including academic research and education. The course project qualifies.
- **No redistribution (7B):** the data may not be transmitted, published, redistributed, or otherwise made available to anyone who has not accepted the competition rules, and reasonable measures must keep it from such people.
- **Derived artifacts:** the rules do not address trained models, aggregate metrics, or figures explicitly. The winners' open-source license applies only to prize winners.

Project rules that follow (interpretation, not legal advice):

| Artifact | Allowed | Reason |
|---|---|---|
| Code, aggregate metrics, plots of aggregates | Yes | Contain no rows of Competition Data |
| Raw or sample rows (repo, slides, screenshots, demos) | No | Redistribution under 7B; course audiences have not accepted the rules |
| Sending rows to third-party APIs or hosted models | No | Transmission under 7B; blocks the [Jev proposal](../proposals/jev-experiment.md) on real data |
| Trained model files | Not published | Not addressed by the rules; models can memorize rows, so kept local |
| Training on hosted compute (for example, notebooks in the cloud) | Avoid; train locally | Copies data to a third party's infrastructure; not addressed explicitly |

Downloading requires a Kaggle account that has accepted the rules and an API token configured by the owner; the token is never committed.

## Structure

As described on the competition data page (summary reviewed during planning; recheck when downloading):

- Two tables, **transaction** and **identity**, joined on `TransactionID`. Not every transaction has an identity row.
- Label: `isFraud` in the training transaction table.
- `TransactionDT` is a time delta from an undisclosed reference datetime, not a real timestamp.
- Many columns are anonymized or masked; their business meaning is not published.
- Measured from the downloaded files on 2026-11-08 (files dated 2019-12-11):

| Item | Value |
|---|---|
| `train_transaction.csv` | 590,540 rows; 683,351,067 bytes |
| `train_identity.csv` | 144,233 rows; 26,529,680 bytes (24% of transactions have identity) |
| Fraud rate | 3.50% |
| `TransactionDT` span | 86,400 to 15,811,131 seconds (about 182 days, if the unit is seconds) |
| Features after prepare | 400 numeric, 31 text columns encoded as categories |

- Split with `ml/params.yaml` (0.70 / 0.15 / 0.15), from the prepare manifest:

| Range | Rows | Fraud rate | Starts at `TransactionDT` |
|---|---|---|---|
| Train | 413,378 | 3.52% | 86,400 |
| Validation | 88,581 | 3.43% | 10,438,003 |
| Test | 88,581 | 3.48% | 13,151,880 |

- Some numeric columns (for example `card1`, `addr1`) are codes rather than quantities; they pass through as numbers for now, and their treatment is a feature engineering decision (M3).

## How this project uses it

- **Labeled data only.** The competition test set has no public labels, so evaluation uses a held-out, later time range of the labeled training data.
- **Time-based split.** Train, validation, and test are consecutive `TransactionDT` ranges; cuts fall on timestamp changes, and boundaries are recorded in the prepare stage's `manifest.json` (fractions in `ml/params.yaml`).
- **Replay as traffic.** Rows are replayed in `TransactionDT` order as simulated requests; replay speed is a parameter.

## Leakage rules

1. Preprocessing (encoders, imputers, scalers) is fitted on the training range only.
2. Any historical or velocity feature for a transaction uses only transactions strictly before it; ties on `TransactionDT` are ordered by `TransactionID` and that rule is tested.
3. No feature uses the label or information available only after the decision time.
4. Hyperparameters are tuned on validation; the test range is used once per reported result.

## Feature availability assumptions

Replaying a full historical row shows that the serving path works; it does not prove each field would exist at real authorization time. The feature contract lists, per feature group, whether it is assumed available at decision time, and the online velocity features (Redis) are defined so that offline and online computation match.

## Known limitations

- Historical, anonymized data; not live traffic and not representative of any specific issuer.
- Masked columns prevent business interpretation; this project does not invent meanings for them.
- Card and address fields are not a verified customer identity; no account-level linking is claimed.
- No real chargeback costs or review capacity; both are scenario parameters.
- Labels are given at once; delayed label arrival (scope S-16) is simulated.

## Change log

- 2026-11-05: Initial draft; terms of use unverified.
- 2026-11-05: Terms of use verified from competition rules section 7; artifact rules added.
- 2026-11-08: Row counts, file sizes, and split figures recorded from the first prepare run.
