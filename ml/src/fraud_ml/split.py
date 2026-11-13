"""Split rows into consecutive train, validation, and test time ranges.

Why by time: in production a model is trained on the past and scores the
future. A random split would let the model see later fraud patterns during
training and overstate its quality.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from fraud_ml.config import SplitParams
from fraud_ml.data import ID, TIME


@dataclass(frozen=True)
class TimeSplit:
    """The three ranges and the boundaries that define them.

    ``valid_start`` and ``test_start`` are ``TransactionDT`` values: a row is
    in train if its time is below ``valid_start``, in validation if below
    ``test_start``, and in test otherwise.
    """

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    valid_start: int
    test_start: int


def time_boundary(times: pd.Series, row_index: int) -> int:
    """Return the time value where a range starting at ``row_index`` should begin.

    If the row at ``row_index`` shares its timestamp with the row before it,
    the cut moves forward to the next distinct timestamp, so no
    ``TransactionDT`` value appears in two ranges.

    Raises:
        ValueError: If no later timestamp exists to start the range.
    """
    row_index = min(row_index, len(times) - 1)
    value = times.iloc[row_index]
    if row_index == 0 or times.iloc[row_index - 1] != value:
        return int(value)
    later = times[times > value]
    if later.empty:
        raise ValueError("no timestamp left to start a split range")
    return int(later.iloc[0])


def time_split(df: pd.DataFrame, params: SplitParams) -> TimeSplit:
    """Split ``df`` into consecutive time ranges.

    Rows are ordered by ``TransactionDT`` with ties broken by
    ``TransactionID`` (dataset card, leakage rule 2), then cut at the
    configured fractions, adjusted to timestamp boundaries.

    Args:
        df: Validated table.
        params: Split fractions.

    Returns:
        The three ranges with their boundaries.

    Raises:
        ValueError: If any range would be empty.
    """
    ordered = df.sort_values([TIME, ID], kind="stable").reset_index(drop=True)
    times = ordered[TIME]
    n = len(ordered)
    valid_start = time_boundary(times, int(n * params.train_fraction))
    test_start = time_boundary(
        times, max(int(n * (params.train_fraction + params.valid_fraction)), 1)
    )

    train = ordered[times < valid_start]
    valid = ordered[(times >= valid_start) & (times < test_start)]
    test = ordered[times >= test_start]
    if train.empty or valid.empty or test.empty:
        raise ValueError("a split range is empty; adjust fractions or check the data")
    return TimeSplit(train, valid, test, valid_start, test_start)
