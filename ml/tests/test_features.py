import numpy as np
import pandas as pd
import pytest

from fraud_ml.config import FeatureParams
from fraud_ml.features import add_features

PARAMS = FeatureParams(enabled=True, group_keys=[["card4"], ["card4", "ProductCD"]])


def fe(frame: pd.DataFrame) -> pd.DataFrame:
    out = add_features(frame, PARAMS)
    return out[["TransactionID", *[c for c in out.columns if c.startswith("fe_")]]]


def test_known_values_for_a_small_group():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [100, 160, 400],
            "isFraud": [0, 0, 0],
            "TransactionAmt": [10.0, 30.0, 40.0],
            "card4": ["visa", "visa", "visa"],
            "ProductCD": ["W", "W", "C"],
        }
    )
    out = fe(df).set_index("TransactionID")
    assert out["fe_card4_prior_count"].tolist() == [0, 1, 2]
    assert np.isnan(out.loc[1, "fe_card4_prior_amount_mean"])
    assert out.loc[3, "fe_card4_prior_amount_mean"] == pytest.approx(20.0)
    assert out.loc[3, "fe_card4_amount_ratio"] == pytest.approx(2.0)
    assert out.loc[2, "fe_card4_seconds_since_previous"] == 60
    assert out.loc[3, "fe_card4_ProductCD_prior_count"] == 0  # first "C"


def test_features_never_depend_on_later_rows(raw_tables):
    """Leakage rule 2: changing or removing the future leaves the past unchanged."""
    df = raw_tables[0]
    full = fe(df)
    cut = df["TransactionDT"].quantile(0.6)
    past_only = fe(df[df["TransactionDT"] <= cut])
    later = df["TransactionDT"] > cut
    altered_future = fe(df.assign(TransactionAmt=np.where(later, 1e6, df["TransactionAmt"])))

    n = len(past_only)
    pd.testing.assert_frame_equal(full.iloc[:n], past_only)
    pd.testing.assert_frame_equal(altered_future.iloc[:n], past_only)


def test_ties_count_only_lower_transaction_ids_as_earlier():
    df = pd.DataFrame(
        {
            "TransactionID": [7, 5, 6],
            "TransactionDT": [100, 100, 100],
            "isFraud": [0, 0, 0],
            "TransactionAmt": [1.0, 2.0, 3.0],
            "card4": ["visa"] * 3,
            "ProductCD": ["W"] * 3,
        }
    )
    out = fe(df).set_index("TransactionID")
    assert out["fe_card4_prior_count"].to_dict() == {5: 0, 6: 1, 7: 2}


def test_labels_are_not_used(raw_tables):
    df = raw_tables[0]
    pd.testing.assert_frame_equal(fe(df), fe(df.assign(isFraud=1 - df["isFraud"])))


def test_missing_key_gives_no_group_features(raw_tables):
    out = add_features(raw_tables[0], PARAMS)
    no_card = out["card4"].isna()
    assert no_card.any()
    assert out.loc[no_card, "fe_card4_prior_count"].isna().all()


def test_disabled_only_sorts(raw_tables):
    out = add_features(raw_tables[0], FeatureParams(enabled=False, group_keys=[]))
    assert not any(c.startswith("fe_") for c in out.columns)
    assert out["TransactionDT"].is_monotonic_increasing


def test_unknown_group_key_fails(raw_tables):
    with pytest.raises(KeyError):
        add_features(raw_tables[0], FeatureParams(enabled=True, group_keys=[["nope"]]))


CLIENT = FeatureParams(
    enabled=True,
    group_keys=[["card4", "fe_D1_start"]],
    day_offsets=["D1"],
    client_key=["card4", "fe_D1_start"],
    client_means=["C1"],
    client_nunique=["ProductCD"],
)


def with_client_columns(df: pd.DataFrame, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    day = df["TransactionDT"] // 86400
    start = rng.integers(0, 3, size=len(df))  # three possible card start days
    c1 = rng.integers(1, 5, size=len(df)).astype(float)
    c1[::7] = np.nan
    return df.assign(D1=day - start, C1=c1)


def test_day_offset_and_client_features_known_values():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [86_400, 2 * 86_400, 3 * 86_400, 3 * 86_400 + 5],
            "isFraud": [0, 0, 0, 0],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "card4": ["visa"] * 4,
            "ProductCD": ["W", "C", "W", "R"],
            "D1": [0, 1, 2, 0],  # rows 1-3 share start day 1; row 4 starts on day 3
            "C1": [2.0, np.nan, 4.0, 9.0],
        }
    )
    out = add_features(df, CLIENT).set_index("TransactionID")
    assert out["fe_D1_start"].tolist() == [1, 1, 1, 3]
    assert out["fe_card4_fe_D1_start_prior_count"].tolist() == [0, 1, 2, 0]
    means = out["fe_client_C1_prior_mean"]
    assert np.isnan(means[1]) and means[2] == 2.0 and means[3] == 2.0 and np.isnan(means[4])
    assert out["fe_client_ProductCD_prior_nunique"].tolist() == [0, 1, 2, 0]


def test_client_features_never_depend_on_later_rows(raw_tables):
    df = with_client_columns(raw_tables[0])
    cols = lambda o: o[[c for c in o.columns if c.startswith("fe_")]]  # noqa: E731
    full = cols(add_features(df, CLIENT))
    cut = df["TransactionDT"].quantile(0.6)
    later = df["TransactionDT"] > cut
    past_only = cols(add_features(df[~later], CLIENT))
    altered = cols(
        add_features(
            df.assign(
                C1=np.where(later, 99.0, df["C1"]), ProductCD=np.where(later, "Z", df["ProductCD"])
            ),
            CLIENT,
        )
    )
    n = len(past_only)
    pd.testing.assert_frame_equal(full.iloc[:n], past_only)
    pd.testing.assert_frame_equal(altered.iloc[:n], past_only)
    pd.testing.assert_frame_equal(
        cols(add_features(df.assign(isFraud=1 - df["isFraud"]), CLIENT)), full
    )
