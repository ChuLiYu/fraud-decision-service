# Tuning guide

What each parameter in [`ml/params.yaml`](../../ml/params.yaml) does, what changes when you move it, and what to watch. Values live only in `params.yaml`; this guide explains them and grows with each milestone.

Reading version (zh-TW): [tuning-guide.zh-TW.md](tuning-guide.zh-TW.md)

Any change alters the manifest's params hash, so rerun `make prepare` before training.

## split

| Parameter | Effect | Raising it | Lowering it | Watch |
|---|---|---|---|---|
| `train_fraction` | Share of earliest rows used to fit preprocessing and models | More history to learn from; validation and test move later and shrink | Less training data; more rows to evaluate | Fraud count in each range (manifest `split.ranges`); few frauds make metrics noisy |
| `valid_fraction` | Share used for tuning and threshold choice | Steadier tuning signal; smaller test range | Noisier tuning | Test must keep enough frauds for a stable PR-AUC |

Cuts move to the next timestamp change so no `TransactionDT` value spans two ranges. Ranges are not shuffled: that is the point of a time split.

## preprocess

| Parameter | Effect | Raising it | Lowering it | Watch |
|---|---|---|---|---|
| `min_category_count` | Category values rarer than this in training become unknown (`-1`) | Fewer, more reliable categories; less memorization of rare values | More categories, including rare ones the model may overfit | Number of categories per column in `preprocessor.json`; gap between validation and test results |

## Planned

Model hyperparameters (M2), feature engineering windows (M3), and decision thresholds and review rates (M4) are added here when those stages exist.
