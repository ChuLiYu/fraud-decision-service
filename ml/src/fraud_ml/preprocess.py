"""Preprocessing shared by training and (later) the inference service.

The preprocessor is fitted on the training range only (dataset card,
leakage rule 1) and saved as JSON rather than pickle, so it is readable,
diffable, and loadable from any service without executing code.

What it does:

* Numeric columns pass through as ``float32``; missing values stay missing,
  because LightGBM handles them natively and the baseline imputes inside its
  own model pipeline.
* Text columns become integer codes from a vocabulary built on the training
  range. Values that are missing, unseen in training, or rarer than
  ``min_category_count`` all map to ``-1``, which LightGBM treats as missing.

``TransactionID``, ``TransactionDT``, and ``isFraud`` are carried through but
are not features: an absolute timestamp would let the model learn the
period rather than the pattern.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from fraud_ml.data import ID, LABEL, TIME

FORMAT_VERSION = 1
UNKNOWN = -1
_NOT_FEATURES = (ID, TIME, LABEL)


@dataclass
class Preprocessor:
    """Fitted column lists and category vocabularies."""

    min_category_count: int
    numeric: list[str] = field(default_factory=list)
    categorical: dict[str, list[str]] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        """Feature columns in the order models receive them."""
        return [*self.numeric, *self.categorical]

    def fit(self, train: pd.DataFrame) -> Preprocessor:
        """Learn column types and vocabularies from the training range only.

        Vocabulary order is by descending training count, then by value, so
        refitting on the same data always yields the same codes.
        """
        self.numeric, self.categorical = [], {}
        for column in train.columns:
            if column in _NOT_FEATURES:
                continue
            if pd.api.types.is_numeric_dtype(train[column]):
                self.numeric.append(column)
                continue
            counts = train[column].dropna().astype(str).value_counts()
            kept = counts[counts >= self.min_category_count]
            ordered = sorted(kept.items(), key=lambda item: (-item[1], item[0]))
            self.categorical[column] = [value for value, _ in ordered]
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the fitted preprocessing; never learns from ``df``.

        Raises:
            KeyError: If a fitted feature column is missing from ``df``.
        """
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            raise KeyError(f"columns missing at transform time: {missing[:10]}")
        out = {c: df[c] for c in _NOT_FEATURES if c in df.columns}
        for column in self.numeric:
            out[column] = df[column].astype("float32")
        for column, vocabulary in self.categorical.items():
            codes = {value: index for index, value in enumerate(vocabulary)}
            mapped = df[column].astype("string").map(codes)
            out[column] = mapped.fillna(UNKNOWN).astype("int32")
        return pd.DataFrame(out, index=df.index)

    def save(self, path: Path) -> None:
        """Write the fitted state as JSON."""
        state = {
            "format_version": FORMAT_VERSION,
            "min_category_count": self.min_category_count,
            "numeric": self.numeric,
            "categorical": self.categorical,
        }
        path.write_text(json.dumps(state, indent=1, sort_keys=True))

    @classmethod
    def load(cls, path: Path) -> Preprocessor:
        """Read a fitted preprocessor written by :meth:`save`.

        Raises:
            ValueError: If the file was written by an incompatible version.
        """
        state = json.loads(path.read_text())
        if state.get("format_version") != FORMAT_VERSION:
            raise ValueError(f"unsupported preprocessor format: {state.get('format_version')}")
        return cls(
            min_category_count=state["min_category_count"],
            numeric=state["numeric"],
            categorical=state["categorical"],
        )
