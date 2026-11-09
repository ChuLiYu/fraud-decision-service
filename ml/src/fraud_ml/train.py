"""Stage 3: train the baseline and LightGBM models from prepare's artifacts.

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

Outputs go to ``data/models`` (local only; dataset card) with a manifest
that records the parameters and the exact data files used.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from fraud_ml import manifest
from fraud_ml.config import Params, load_params
from fraud_ml.data import LABEL
from fraud_ml.prepare import PARAM_SECTIONS as DATA_SECTIONS
from fraud_ml.preprocess import Preprocessor

MODEL_SECTIONS = (*DATA_SECTIONS, "seed", "baseline", "lightgbm")
BASELINE_FILE = "baseline.joblib"
LIGHTGBM_FILE = "lightgbm.txt"


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


def fit_baseline(train: pd.DataFrame, pre: Preprocessor, params: Params) -> Pipeline:
    """Fit the logistic regression baseline on the training range."""
    columns = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        # keep_empty_features: a column empty in train stays as zeros,
                        # so the feature count never changes between ranges.
                        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
                        # Quantiles instead of standardizing: values beyond the
                        # training range map to the extreme quantile, so a linear
                        # model cannot be pushed to p = 1.0 by an unseen outlier
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
    model = Pipeline(
        [
            ("columns", columns),
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

    processed = params.data.processed_dir
    return manifest.write(
        out,
        params_hash=models_hash(params),
        inputs=[processed / "train.parquet", processed / "valid.parquet"],
        outputs=[out / BASELINE_FILE, out / LIGHTGBM_FILE],
        details={
            "baseline": {
                "train_seconds": round(baseline_seconds, 1),
                "iterations": int(baseline.named_steps["classifier"].n_iter_[0]),
            },
            "lightgbm": {
                "train_seconds": round(lightgbm_seconds, 1),
                "best_iteration": booster.best_iteration,
            },
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
