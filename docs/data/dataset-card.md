# Dataset card: IEEE-CIS Fraud Detection

Status: draft. Items marked **unverified** must be checked against the Kaggle competition pages before phase 1 code or any publication.

## Source

- Kaggle competition "IEEE-CIS Fraud Detection": https://www.kaggle.com/c/ieee-fraud-detection
- Access requires a Kaggle account that has accepted the competition rules. Data is downloaded by script into an ignored `data/` directory and never committed.

## Terms of use (unverified)

The competition rules page could not be read automatically on 2026-11-05 (the page requires a browser session). Before relying on the data, the owner records here:

- Permitted uses after the competition (education, research, non-commercial).
- Whether derived artifacts may be published: trained models, aggregated metrics, sample rows, screenshots, demos.
- Attribution requirements.

Until then: publish code and aggregate metrics only; no raw or sample rows, no trained model files.

## Structure

As described on the competition data page (summary reviewed during planning; recheck when downloading):

- Two tables, **transaction** and **identity**, joined on `TransactionID`. Not every transaction has an identity row.
- Label: `isFraud` in the training transaction table.
- `TransactionDT` is a time delta from an undisclosed reference datetime, not a real timestamp.
- Many columns are anonymized or masked; their business meaning is not published.
- Row counts and file sizes: record from the downloaded files (not stated here to avoid unverified figures).

## How this project uses it

- **Labeled data only.** The competition test set has no public labels, so evaluation uses a held-out, later time range of the labeled training data.
- **Time-based split.** Train, validation, and test are consecutive `TransactionDT` ranges; boundaries are recorded with the split code.
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
