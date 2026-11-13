import dataclasses
import json

import lightgbm as lgb

from fraud_ml.bundle import build, serving_params, synthetic_transactions
from fraud_ml.config import load_params
from fraud_ml.preprocess import Preprocessor


def test_serving_params_drop_history_features(workspace):
    params = serving_params(load_params(workspace / "params.yaml"))
    assert params.features.group_keys == []
    assert params.features.client_key == []


def test_demo_bundle_loads_and_scores(workspace, tmp_path):
    params = load_params(workspace / "params.yaml")
    params = dataclasses.replace(
        params, features=dataclasses.replace(params.features, day_offsets=["D1"])
    )
    out = tmp_path / "bundle"
    build(synthetic_transactions(n=3_000), params, out, "demo")

    info = json.loads((out / "bundle.json").read_text())
    policy = json.loads((out / "policy.json").read_text())
    assert info["kind"] == "demo" and info["model_version"].startswith("demo-")
    assert "TransactionAmt" in info["raw_fields"] and "isFraud" not in info["raw_fields"]
    assert policy["t_review"] is None or 0 <= policy["t_review"] <= 1
    pre = Preprocessor.load(out / "preprocessor.json")
    booster = lgb.Booster(model_file=str(out / "lightgbm.txt"))
    assert booster.num_feature() == len(pre.feature_names)
    assert not any(name.startswith("fe_card") for name in pre.feature_names)
