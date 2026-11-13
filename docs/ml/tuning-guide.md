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

## features

Engineered in `ml/src/fraud_ml/features.py` from the transaction and earlier transactions only (leakage rule 2).

| Parameter | Effect | Changing it | Watch |
|---|---|---|---|
| `enabled` | Turns all engineered features on or off | `false` reproduces the M2 feature set, for ablation | Validation PR-AUC in `reports/comparison.md` |
| `group_keys` | Each list defines transactions that share those field values; each adds 4 history features (prior count, prior mean amount, amount ratio, seconds since previous) | More or longer keys: finer groups, more features, fewer earlier transactions per group | Same; also LightGBM feature importance once available |

Groups are shared field values, not verified customers (dataset card).

## preprocess

| Parameter | Effect | Raising it | Lowering it | Watch |
|---|---|---|---|---|
| `min_category_count` | Category values rarer than this in training become unknown (`-1`) | Fewer, more reliable categories; less memorization of rare values | More categories, including rare ones the model may overfit | Number of categories per column in `preprocessor.json`; gap between validation and test results |

## seed

Fixes randomness in LightGBM (row and feature sampling) and the baseline. Changing it and seeing a large metric change means a result is within noise. `make tune` measures this noise directly (see `tuning.seeds` below); compare a difference between settings with the standard deviation across seeds before calling it an improvement.

## baseline (logistic regression)

Numeric features are mapped to normal quantiles learned on the training range, so values beyond that range cannot push the model to extreme scores.

| Parameter | Effect | Raising it | Lowering it | Watch |
|---|---|---|---|---|
| `C` | Inverse regularization strength | Weaker regularization; fits training data more closely | Stronger regularization; smaller weights, simpler model | Validation PR-AUC versus training |
| `max_iter` | Solver iteration limit | Lets the solver converge; slower | Faster; may stop before converging | `baseline.iterations` in the models manifest: equal to `max_iter` means it did not converge |

## lightgbm

| Parameter | Effect | Raising it | Lowering it | Watch |
|---|---|---|---|---|
| `learning_rate` | How much each tree corrects the previous ones | Fewer rounds needed; risk of overshooting | More rounds and time; often slightly better | `best_iteration` in the report |
| `num_leaves` | Maximum leaves per tree (complexity) | Captures more interactions; overfits more easily | Simpler trees; may underfit | Gap between validation and test |
| `min_child_samples` | Minimum rows per leaf | Smoother, more general rules | Rules from fewer rows; can memorize rare cases | Same as above; fraud is rare, so very low values overfit |
| `feature_fraction` | Share of features sampled per tree | Each tree sees more features | More randomness; less reliance on a few strong features | Stability of results across seeds |
| `bagging_fraction`, `bagging_freq` | Row sampling per iteration (`freq` 0 disables) | Less randomness | More randomness and speed | As above |
| `lambda_l2` | L2 penalty on leaf values | Smaller, more conservative leaf scores | Leaves fit training data more closely | Calibration (ECE) and validation PR-AUC |
| `max_rounds` | Upper bound on trees | Allows more rounds if early stopping has not triggered | Caps training time | If `best_iteration` equals `max_rounds`, raise it |
| `early_stopping_rounds` | Rounds without validation PR-AUC improvement before stopping | Tolerates plateaus; slower | Stops sooner; may stop too early | `best_iteration` |

Tune these against the **validation** range only. The test range is for one final result per configuration.

## mlp

The comparison model: a feed-forward neural network on the baseline's inputs (quantile-normalized numeric features plus one-hot categories). Each epoch is one pass over the training range; training stops when validation PR-AUC has not improved for `patience` epochs, and the best epoch's weights are kept.

| Parameter | Effect | Raising it | Lowering it | Watch |
|---|---|---|---|---|
| `hidden_layer_sizes` | Neurons per hidden layer | More capacity; slower; overfits more easily | Closer to a linear model | Gap to the baseline; best epoch |
| `alpha` | L2 penalty on weights | Smoother, more conservative functions | Fits training data more closely | Validation PR-AUC |
| `learning_rate_init` | Adam step size | Faster progress; unstable if too high | Slower, steadier | Validation PR-AUC per epoch |
| `batch_size` | Rows per gradient step | Faster epochs; smoother gradients | Noisier steps; slower | Training time |
| `max_epochs`, `patience` | Upper bound and early-stopping patience | Trains longer | Stops sooner | If the best epoch equals `max_epochs`, raise it |

## tuning

Read only by `make tune`, which trains LightGBM on the train range for each setting, scores the validation range, and writes [`ml/reports/tuning/tuning.md`](../../ml/reports/tuning/tuning.md). It never reads the test range and never edits `params.yaml`: the owner copies the chosen values into the `lightgbm` section, and that commit records the decision.

| Parameter | Effect | Watch |
|---|---|---|
| `grid` | LightGBM parameters and the values to try; every combination is trained once (cost grows multiplicatively) | If the best value sits at the edge of the grid, the optimum may lie beyond it |
| `seeds` | Seeds used to retrain the current and the best grid setting | Standard deviation across seeds: a gap smaller than about two standard deviations is not a reliable improvement |

Two limits apply to the scores. Early stopping also uses the validation range, and picking the best of several settings on the same range adds selection bias, so validation scores are optimistic. Use them to rank settings; the test range of the next evaluation run gives the unbiased result.

## fusion

Read only by `make fuse`. The protocol is fixed before the run (dataset card, leakage rule 5): fusion is fitted on the first half of the validation range, chosen on the second half, and reported once on test.

| Parameter | Effect |
|---|---|
| `blend_weights` | LightGBM's share values tried for the logit blend of LightGBM and the MLP; `1.0` is LightGBM alone |
| `min_gain` | Test PR-AUC gain over LightGBM alone required to adopt fusion; the chosen method must also win every test time slice. Set it before running, never after seeing the result |

## evaluation

These change how results are summarized, not the models.

| Parameter | Effect |
|---|---|
| `review_rates` | Shares of transactions sent to review at which precision and recall are reported. Review capacity is a scenario parameter (dataset card), so several rates are shown |
| `calibration_bins` | Number of equal-count bins in the reliability table. More bins show more detail but each is noisier |
| `time_slices` | Number of equal-row slices of each range in time order. More slices reveal drift in more detail, with fewer frauds per slice |

## decision

Read only by `make decide`. Costs are scenario assumptions, not measured figures (dataset card), so the stage reports every policy under each named scenario and shows which one is cheapest in each; a conclusion that holds only in one scenario is weak. Every policy is fitted on the validation range and applied unchanged to the test range.

Each entry under `scenarios` (one must be named `base`):

| Parameter | Effect |
|---|---|
| `review_cost_cents` | Cost of one manual review. Higher values make policies review less |
| `fraud_loss_rate` | Share of an approved fraud's amount that is lost. Higher values push toward more review and decline |
| `false_decline_rate` | Share of a declined legitimate amount that is lost (margin) |
| `false_decline_fixed_cents` | Fixed cost per false decline (customer friction, churn). Without it, declining small legitimate transactions looks almost free and policies decline far too much |

Shared settings:

| Parameter | Effect |
|---|---|
| `review_capacities` | Most transactions the review team can handle, as a share; one policy is fitted per value |
| `max_decline_rate` | Upper bound of the threshold policies' decline search; the report lists any policy that stopped at it |
| `threshold_grid` | Candidate cut points per threshold; more points find slightly better thresholds, slower |
| `bootstrap` | Resamples for the 95% intervals on test cost differences; more resamples give steadier intervals |
