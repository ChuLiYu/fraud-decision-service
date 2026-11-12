"""Load and validate ``params.yaml``.

All tunable values live in ``params.yaml``; code reads them only through
:func:`load_params` so a typo or out-of-range value fails fast instead of
silently producing a different experiment.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DataParams:
    """Locations of raw and processed data, relative to the params file."""

    raw_dir: Path
    processed_dir: Path
    models_dir: Path
    reports_dir: Path


@dataclass(frozen=True)
class SplitParams:
    """Shares of rows for the train and validation time ranges."""

    train_fraction: float
    valid_fraction: float

    def __post_init__(self) -> None:
        """Reject fractions that would leave an empty or negative range."""
        if not (0 < self.train_fraction < 1 and 0 < self.valid_fraction < 1):
            raise ValueError("split fractions must be between 0 and 1")
        if self.train_fraction + self.valid_fraction >= 1:
            raise ValueError("train_fraction + valid_fraction must leave rows for test")


@dataclass(frozen=True)
class PreprocessParams:
    """Settings for fitting the shared preprocessing on the training range."""

    min_category_count: int

    def __post_init__(self) -> None:
        """Reject a non-positive minimum count."""
        if self.min_category_count < 1:
            raise ValueError("min_category_count must be at least 1")


@dataclass(frozen=True)
class FeatureParams:
    """Time-safe feature engineering settings (see fraud_ml.features)."""

    enabled: bool
    group_keys: list[list[str]]


@dataclass(frozen=True)
class BaselineParams:
    """Logistic regression settings."""

    C: float
    max_iter: int


@dataclass(frozen=True)
class LightGBMParams:
    """LightGBM settings; see docs/ml/tuning-guide.md for each one."""

    learning_rate: float
    num_leaves: int
    min_child_samples: int
    feature_fraction: float
    bagging_fraction: float
    bagging_freq: int
    lambda_l2: float
    max_rounds: int
    early_stopping_rounds: int


@dataclass(frozen=True)
class EvaluationParams:
    """How results are summarized."""

    review_rates: list[float]
    calibration_bins: int
    time_slices: int

    def __post_init__(self) -> None:
        """Reject review rates outside (0, 1) and non-positive counts."""
        if not all(0 < rate < 1 for rate in self.review_rates):
            raise ValueError("review_rates must be between 0 and 1")
        if self.calibration_bins < 2 or self.time_slices < 1:
            raise ValueError("calibration_bins must be >= 2 and time_slices >= 1")


@dataclass(frozen=True)
class Params:
    """All pipeline parameters."""

    data: DataParams
    split: SplitParams
    preprocess: PreprocessParams
    features: FeatureParams
    seed: int
    baseline: BaselineParams
    lightgbm: LightGBMParams
    evaluation: EvaluationParams

    def stage_hash(self, *sections: str) -> str:
        """Return a stable SHA-256 of the named sections.

        Paths are excluded on purpose: moving the data directory must not
        invalidate artifacts, but changing a split fraction must.
        """
        payload = {}
        for name in sections:
            value = getattr(self, name)
            payload[name] = asdict(value) if hasattr(value, "__dataclass_fields__") else value
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode()).hexdigest()


def load_params(path: Path) -> Params:
    """Read ``params.yaml`` and resolve data paths relative to its directory.

    Args:
        path: Location of the params file.

    Returns:
        Validated parameters.

    Raises:
        ValueError: If a value is out of range.
        KeyError: If a required key is missing.
    """
    raw = yaml.safe_load(path.read_text())
    base = path.resolve().parent
    return Params(
        data=DataParams(**{key: (base / value).resolve() for key, value in raw["data"].items()}),
        split=SplitParams(**raw["split"]),
        preprocess=PreprocessParams(**raw["preprocess"]),
        features=FeatureParams(**raw["features"]),
        seed=int(raw["seed"]),
        baseline=BaselineParams(**raw["baseline"]),
        lightgbm=LightGBMParams(**raw["lightgbm"]),
        evaluation=EvaluationParams(**raw["evaluation"]),
    )
