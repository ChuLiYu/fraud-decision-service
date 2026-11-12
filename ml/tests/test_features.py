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
