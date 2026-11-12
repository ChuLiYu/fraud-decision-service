"""Time-safe feature engineering.

Every feature here can be computed at decision time from the transaction
itself and from **earlier** transactions only (dataset card, leakage rule
2). Labels are never used. Earlier means: smaller ``TransactionDT``, or the
same ``TransactionDT`` and a smaller ``TransactionID``.

Because no feature looks ahead, features are computed once over the whole
labeled history in time order, before the split. A test-range transaction
therefore sees earlier validation and test transactions, as a live system
would see all earlier traffic.

Feature groups:

* **Row features:** log amount, cents part of the amount (a common pattern
  difference between merchants and fraud tooling), hour of day, day of
  week, and the number of missing raw fields.
* **History features per group key:** for transactions sharing the values
  of ``group_keys`` (for example ``card1`` and ``addr1``): how many earlier
  transactions the group has, the mean earlier amount, this amount relative
  to that mean, and seconds since the group's previous transaction.

Group keys are combinations of masked card and address fields. They are
not verified customer identity (dataset card): a group is "transactions
sharing these field values", nothing more.

Hour and weekday assume ``TransactionDT`` is in seconds, which the
competition does not state; the dataset card records this as an
assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fraud_ml.config import FeatureParams
from fraud_ml.data import ID, LABEL, TIME

AMOUNT = "TransactionAmt"


def _group_name(keys: list[str]) -> str:
    return "_".join(keys)


def add_features(df: pd.DataFrame, params: FeatureParams) -> pd.DataFrame:
    """Return ``df`` in time order with engineered columns added.

    Args:
        df: Validated raw table (any row order).
        params: Feature settings.

    Returns:
        A new frame sorted by ``TransactionDT`` then ``TransactionID``.

    Raises:
        KeyError: If a group key or the amount column is missing.
    """
    ordered = df.sort_values([TIME, ID], kind="stable").reset_index(drop=True)
    if not params.enabled:
        return ordered

    raw_columns = [c for c in ordered.columns if c not in (ID, TIME, LABEL)]
    amount = ordered[AMOUNT].astype("float64")
    new = {
        "fe_amount_log": np.log1p(amount),
        "fe_amount_cents": (amount - np.floor(amount)).round(3),
        "fe_hour": (ordered[TIME] // 3600) % 24,
        "fe_weekday": (ordered[TIME] // 86400) % 7,
        "fe_missing_count": ordered[raw_columns].isna().sum(axis=1),
    }

    for keys in params.group_keys:
        missing = [k for k in keys if k not in ordered.columns]
        if missing:
            raise KeyError(f"group key columns not found: {missing}")
        name = _group_name(keys)
        # Rows with any missing key value belong to no group: features stay NaN.
        has_key = ordered[keys].notna().all(axis=1)
        frame = ordered.loc[has_key, [*keys, TIME]].assign(_amount=amount[has_key])
        grouped = frame.groupby(keys, sort=False)

        prior_count = grouped.cumcount()  # earlier rows in the group, in time order
        prior_sum = grouped["_amount"].cumsum() - frame["_amount"]
        prior_mean = (prior_sum / prior_count).where(prior_count > 0)
        since_previous = frame[TIME] - grouped[TIME].shift(1)

        new[f"fe_{name}_prior_count"] = prior_count.reindex(ordered.index)
        new[f"fe_{name}_prior_amount_mean"] = prior_mean.reindex(ordered.index)
        new[f"fe_{name}_amount_ratio"] = (amount / prior_mean).reindex(ordered.index)
        new[f"fe_{name}_seconds_since_previous"] = since_previous.reindex(ordered.index)

    return pd.concat([ordered, pd.DataFrame(new, index=ordered.index).astype("float64")], axis=1)
