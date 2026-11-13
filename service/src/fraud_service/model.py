"""Load a serving bundle and score one transaction.

The bundle is written by ``fraud_ml.bundle``. Scoring reuses ``fraud_ml``
feature and preprocessing code, so a request is transformed exactly as the
training rows were.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from fraud_ml.bundle import BUNDLE_FILE, MODEL_FILE, POLICY_FILE, PREPROCESSOR_FILE
from fraud_ml.config import FeatureParams
from fraud_ml.data import ID, TIME
from fraud_ml.features import AMOUNT, add_features
from fraud_ml.preprocess import Preprocessor

SUPPORTED_FORMAT = 1


@dataclass(frozen=True)
class Policy:
    """Score thresholds; ``None`` means the outcome is never used."""

    name: str
    version: str
    t_review: float | None
    t_decline: float | None

    def outcome(self, score: float) -> tuple[str, str]:
        """Outcome and reason code for a valid score."""
        if self.t_decline is not None and score >= self.t_decline:
            return "decline", "score_at_or_above_decline_threshold"
        if self.t_review is not None and score >= self.t_review:
            return "review", "score_at_or_above_review_threshold"
        return "approve", "score_below_review_threshold"


@dataclass(frozen=True)
class Bundle:
    """Everything needed to score a request."""

    kind: str
    model_version: str
    raw_fields: list[str]
    features: FeatureParams
    preprocessor: Preprocessor
    booster: lgb.Booster
    policy: Policy

    @property
    def request_fields(self) -> list[str]:
        """Optional raw fields a request may carry (time and amount are separate)."""
        return [f for f in self.raw_fields if f not in (TIME, AMOUNT)]

    def score(self, transaction_id: int, seconds: int, amount: float, fields: dict) -> float:
        """Fraud score in [0, 1] for one transaction.

        Raises:
            ValueError: If the model returns a non-finite score.
        """
        row = dict.fromkeys(self.raw_fields, np.nan)
        row.update(fields)
        row.update({ID: transaction_id, TIME: seconds, AMOUNT: amount})
        frame = pd.DataFrame([row])
        # Same column types as training: categoricals as strings, numerics as floats.
        for column in self.raw_fields:
            if column not in self.preprocessor.categorical and column != TIME:
                frame[column] = pd.to_numeric(frame[column], errors="raise").astype("float64")
        features = self.preprocessor.transform(add_features(frame, self.features))
        score = float(self.booster.predict(features[self.preprocessor.feature_names])[0])
        if not math.isfinite(score) or not 0.0 <= score <= 1.0:
            raise ValueError(f"invalid model score: {score}")
        return score


def load(directory: Path) -> Bundle:
    """Read a bundle directory; fails loudly if anything is missing or incompatible."""
    info = json.loads((directory / BUNDLE_FILE).read_text())
    if info.get("format_version") != SUPPORTED_FORMAT:
        raise ValueError(f"unsupported bundle format: {info.get('format_version')}")
    policy = json.loads((directory / POLICY_FILE).read_text())
    return Bundle(
        kind=info["kind"],
        model_version=info["model_version"],
        raw_fields=info["raw_fields"],
        features=FeatureParams(**info["features"]),
        preprocessor=Preprocessor.load(directory / PREPROCESSOR_FILE),
        booster=lgb.Booster(model_file=str(directory / MODEL_FILE)),
        policy=Policy(
            name=policy["name"],
            version=info["policy_version"],
            t_review=policy["t_review"],
            t_decline=policy["t_decline"],
        ),
    )
