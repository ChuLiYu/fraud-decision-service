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

* **Day offsets** (``day_offsets``): masked ``D`` columns count days since
  some earlier event, so the same event drifts by one each day. Subtracting
  them from the transaction day gives a fixed date (``fe_<D>_start``), for
  example the day the card was first seen for ``D1``. This and the client key
  below follow the IEEE-CIS 1st-place solution (Deotte and Yakovlev, 2019).
* **Client key** (``client_key``, for example ``card1``, ``addr1``,
  ``fe_D1_start``): a closer proxy for one account than card fields alone.
  For each transaction, the mean of ``client_means`` columns over the
  key's earlier transactions, and how many distinct values of each
  ``client_nunique`` column the key has shown before. The 1st-place solution
  aggregated over all rows, including later ones, and averaged predictions
  per key afterwards; both look ahead and are not used here.

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

    day = ordered[TIME] // 86400
    for column in params.day_offsets:
        new[f"fe_{column}_start"] = day - ordered[column]
    # Engineered columns may serve as group keys, so expose them for grouping.
    keyed = ordered.assign(**{k: v for k, v in new.items() if k.endswith("_start")})

    for keys in params.group_keys:
        missing = [k for k in keys if k not in keyed.columns]
        if missing:
            raise KeyError(f"group key columns not found: {missing}")
        name = _group_name(keys)
        # Rows with any missing key value belong to no group: features stay NaN.
        has_key = keyed[keys].notna().all(axis=1)
        frame = keyed.loc[has_key, [*keys, TIME]].assign(_amount=amount[has_key])
        grouped = frame.groupby(keys, sort=False)

        prior_count = grouped.cumcount()  # earlier rows in the group, in time order
        prior_sum = grouped["_amount"].cumsum() - frame["_amount"]
        prior_mean = (prior_sum / prior_count).where(prior_count > 0)
        since_previous = frame[TIME] - grouped[TIME].shift(1)

        new[f"fe_{name}_prior_count"] = prior_count.reindex(ordered.index)
        new[f"fe_{name}_prior_amount_mean"] = prior_mean.reindex(ordered.index)
        new[f"fe_{name}_amount_ratio"] = (amount / prior_mean).reindex(ordered.index)
        new[f"fe_{name}_seconds_since_previous"] = since_previous.reindex(ordered.index)

    if params.client_key:
        new.update(_client_features(keyed, params))

    return pd.concat([ordered, pd.DataFrame(new, index=ordered.index).astype("float64")], axis=1)


def _client_features(keyed: pd.DataFrame, params: FeatureParams) -> dict[str, pd.Series]:
    """Past-only means and distinct counts per client key (see module docstring)."""
    keys = params.client_key
    wanted = [*keys, *params.client_means, *params.client_nunique]
    missing = [c for c in wanted if c not in keyed.columns]
    if missing:
        raise KeyError(f"client feature columns not found: {missing}")
    has_key = keyed[keys].notna().all(axis=1)
    frame = keyed.loc[has_key]
    by_key = [frame[k] for k in keys]
    out: dict[str, pd.Series] = {}
    for column in params.client_means:
        value = frame[column].astype("float64")
        present = value.notna().astype("float64")
        filled = value.fillna(0.0)
        prior_sum = filled.groupby(by_key, sort=False).cumsum() - filled
        prior_n = present.groupby(by_key, sort=False).cumsum() - present
        out[f"fe_client_{column}_prior_mean"] = (prior_sum / prior_n).where(prior_n > 0)
    for column in params.client_nunique:
        # A value counts once per key, from its first appearance on; missing is not a value.
        first_seen = (~frame.duplicated([*keys, column]) & frame[column].notna()).astype("int64")
        seen = first_seen.groupby(by_key, sort=False).cumsum() - first_seen
        out[f"fe_client_{column}_prior_nunique"] = seen
    return {name: series.reindex(keyed.index) for name, series in out.items()}
