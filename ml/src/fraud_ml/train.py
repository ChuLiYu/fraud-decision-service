"""Stage 3: train the baseline, LightGBM and MLP models from prepare's artifacts.

Training reads only ``data/processed`` and refuses to start unless
:func:`fraud_ml.manifest.verify` confirms the artifacts match the current
``params.yaml``. It never reads the test range: early stopping uses the
validation range, and the test range is left for one final evaluation.

Models:

* **Baseline:** logistic regression. Numeric features are median-imputed
  and mapped to normal-distribution quantiles learned on the training
  range; categories are one-hot encoded. It shows how much of
  the result a simple linear model already achieves.
* **LightGBM:** gradient-boosted trees, which handle missing values and
  integer category codes natively. Boosting stops when validation PR-AUC
  has not improved for ``early_stopping_rounds`` rounds.
* **MLP (comparison model):** a feed-forward neural network on the same
  inputs as the baseline. It learns smooth functions of normalized
  features instead of axis-aligned splits, so its errors may differ from
  LightGBM's; :mod:`fraud_ml.fusion` tests whether combining them helps.

Outputs go to ``data/models`` (local only; dataset card) with a manifest
that records the parameters and the exact data files used.
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from fraud_ml import manifest
from fraud_ml.config import Params, load_params
from fraud_ml.data import LABEL
from fraud_ml.prepare import PARAM_SECTIONS as DATA_SECTIONS
from fraud_ml.preprocess import Preprocessor

MODEL_SECTIONS = (*DATA_SECTIONS, "seed", "baseline", "lightgbm", "mlp")
BASELINE_FILE = "baseline.joblib"
LIGHTGBM_FILE = "lightgbm.txt"
MLP_FILE = "mlp.joblib"


def models_hash(params: Params) -> str:
    """Hash of every parameter section that shapes the trained models."""
    return params.stage_hash(*MODEL_SECTIONS)


def load_prepared(params: Params, names: tuple[str, ...]) -> tuple[Preprocessor, dict]:
    """Verify prepare's artifacts, then load the preprocessor and ranges.

    Raises:
        ManifestMismatchError: If the artifacts do not match ``params.yaml``.
    """
    out = params.data.processed_dir
    manifest.verify(out, expected_params_hash=params.stage_hash(*DATA_SECTIONS))
    preprocessor = Preprocessor.load(out / "preprocessor.json")
    ranges = {name: pd.read_parquet(out / f"{name}.parquet") for name in names}
    return preprocessor, ranges


def tabular_columns(pre: Preprocessor, params: Params) -> ColumnTransformer:
    """Input transform shared by the baseline and the MLP (unfitted).

    Numeric features are median-imputed and mapped to normal-distribution
    quantiles; categories are one-hot encoded. Both linear models and
    neural networks need inputs on comparable scales, unlike trees.
    """
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        # keep_empty_features: a column empty in train stays as zeros,
                        # so the feature count never changes between ranges.
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        # Quantiles instead of standardizing: values beyond the
                        # training range map to the extreme quantile, so a model
                        # cannot be pushed to p = 1.0 by an unseen outlier
                        # (seen on the real test range; see the regression test).
                        (
                            "scale",
                            QuantileTransformer(
                                n_quantiles=1000,
                                output_distribution="normal",
                                subsample=100_000,
                                random_state=params.seed,
                            ),
                        ),
                    ]
                ),
                pre.numeric,
            ),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), list(pre.categorical)),
        ]
    )


def fit_baseline(train: pd.DataFrame, pre: Preprocessor, params: Params) -> Pipeline:
    """Fit the logistic regression baseline on the training range."""
    model = Pipeline(
        [
            ("columns", tabular_columns(pre, params)),
            (
                "classifier",
                LogisticRegression(
                    C=params.baseline.C,
                    max_iter=params.baseline.max_iter,
                    random_state=params.seed,
                ),
            ),
        ]
    )
    return model.fit(train[pre.feature_names], train[LABEL])


def fit_mlp(
    train: pd.DataFrame, valid: pd.DataFrame, pre: Preprocessor, params: Params
) -> tuple[Pipeline, dict]:
    """Fit the MLP with early stopping on validation PR-AUC.

    scikit-learn's built-in early stopping holds out a random share of the
    training rows and watches accuracy, which is about 96.5% for a model that
    never flags fraud. Instead, each epoch is one ``partial_fit`` pass over
    the training range, scored on the validation range with PR-AUC, the
    same rule LightGBM uses. The weights from the best epoch are kept.

    Returns:
        The fitted pipeline (input transform plus best network) and a
        summary with the best epoch and epochs run.
    """
    settings = params.mlp
    y_train = train[LABEL].to_numpy()
    y_valid = valid[LABEL].to_numpy()
    columns = tabular_columns(pre, params).fit(train[pre.feature_names])
    x_train = columns.transform(train[pre.feature_names])
    x_valid = columns.transform(valid[pre.feature_names])

    network = MLPClassifier(
        hidden_layer_sizes=tuple(settings.hidden_layer_sizes),
        alpha=settings.alpha,
        learning_rate_init=settings.learning_rate_init,
        batch_size=settings.batch_size,
        shuffle=True,
        random_state=params.seed,
    )
    classes = np.array([0, 1])
    best, best_score, best_epoch, epochs_run = None, -np.inf, 0, 0
    for epoch in range(1, settings.max_epochs + 1):
        network.partial_fit(x_train, y_train, classes=classes)
        epochs_run = epoch
        score = average_precision_score(y_valid, network.predict_proba(x_valid)[:, 1])
        if score > best_score:
            best, best_score, best_epoch = copy.deepcopy(network), score, epoch
        elif epoch - best_epoch >= settings.patience:
            break
    model = Pipeline([("columns", columns), ("classifier", best)])
    return model, {"best_epoch": best_epoch, "epochs_run": epochs_run}


def fit_lightgbm(
    train: pd.DataFrame, valid: pd.DataFrame, pre: Preprocessor, params: Params
) -> lgb.Booster:
    """Fit LightGBM with early stopping on the validation range."""
    settings = params.lightgbm
    booster_params = {
        "objective": "binary",
        "metric": "average_precision",  # PR-AUC: suited to rare positives
        "learning_rate": settings.learning_rate,
        "num_leaves": settings.num_leaves,
        "min_child_samples": settings.min_child_samples,
        "feature_fraction": settings.feature_fraction,
        "bagging_fraction": settings.bagging_fraction,
        "bagging_freq": settings.bagging_freq,
        "lambda_l2": settings.lambda_l2,
        "seed": params.seed,
        "deterministic": True,
        "verbosity": -1,
    }
    categorical = list(pre.categorical)

    def dataset(frame: pd.DataFrame, reference: lgb.Dataset | None = None) -> lgb.Dataset:
        return lgb.Dataset(
            frame[pre.feature_names],
            label=frame[LABEL],
            categorical_feature=categorical,
            reference=reference,
        )

    train_set = dataset(train)
    return lgb.train(
        booster_params,
        train_set,
        num_boost_round=settings.max_rounds,
        valid_sets=[dataset(valid, reference=train_set)],
        callbacks=[lgb.early_stopping(settings.early_stopping_rounds, verbose=False)],
    )


def train(params: Params) -> Path:
    """Run the training stage end to end and return the models manifest path."""
    pre, ranges = load_prepared(params, ("train", "valid"))
    out = params.data.models_dir
    out.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    baseline = fit_baseline(ranges["train"], pre, params)
    baseline_seconds = time.perf_counter() - started
    joblib.dump(baseline, out / BASELINE_FILE)

    started = time.perf_counter()
    booster = fit_lightgbm(ranges["train"], ranges["valid"], pre, params)
    lightgbm_seconds = time.perf_counter() - started
    booster.save_model(out / LIGHTGBM_FILE, num_iteration=booster.best_iteration)

    started = time.perf_counter()
    mlp, mlp_summary = fit_mlp(ranges["train"], ranges["valid"], pre, params)
    mlp_seconds = time.perf_counter() - started
    joblib.dump(mlp, out / MLP_FILE)

    processed = params.data.processed_dir
    return manifest.write(
        out,
        params_hash=models_hash(params),
        inputs=[processed / "train.parquet", processed / "valid.parquet"],
        outputs=[out / BASELINE_FILE, out / LIGHTGBM_FILE, out / MLP_FILE],
        details={
            "baseline": {
                "train_seconds": round(baseline_seconds, 1),
                "iterations": int(baseline.named_steps["classifier"].n_iter_[0]),
            },
            "lightgbm": {
                "train_seconds": round(lightgbm_seconds, 1),
                "best_iteration": booster.best_iteration,
            },
            "mlp": {"train_seconds": round(mlp_seconds, 1), **mlp_summary},
        },
    )


def main() -> None:
    """Command-line entry point: ``python -m fraud_ml.train``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    print(f"wrote {train(load_params(args.params))}")


if __name__ == "__main__":
    main()
