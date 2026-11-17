import json
import sys

import lightgbm as lgb

from fraud_ml import bundle
from fraud_ml.bundle import build, demo_params, serving_params, synthetic_transactions
from fraud_ml.config import load_params
from fraud_ml.preprocess import Preprocessor


def test_serving_params_drop_history_features(workspace):
    params = serving_params(load_params(workspace / "params.yaml"))
    assert params.features.group_keys == []
    assert params.features.client_key == []


def test_demo_bundle_loads_and_scores(workspace, tmp_path):
    params = load_params(workspace / "params.yaml")
    params = demo_params(params)
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


def test_saved_preprocessor_keeps_feature_order(workspace, tmp_path):
    """Regression: sorted JSON keys reordered categorical columns after loading."""
    out = tmp_path / "bundle"
    build(
        synthetic_transactions(n=3_000),
        demo_params(load_params(workspace / "params.yaml")),
        out,
        "demo",
    )
    pre = Preprocessor.load(out / "preprocessor.json")
    booster = lgb.Booster(model_file=str(out / "lightgbm.txt"))
    assert booster.feature_name() == pre.feature_names


def test_demo_cli_passes_seed_to_synthetic_rows(workspace, tmp_path, monkeypatch):
    """Demo bundle generation accepts a reproducibility seed without Competition Data."""
    observed = {}

    def synthetic(n=20_000, seed=0):
        observed["seed"] = seed
        return synthetic_transactions(n=100, seed=seed)

    def fake_build(raw, params, out, kind):
        observed["rows"] = len(raw)
        observed["kind"] = kind
        return out / "bundle.json"

    monkeypatch.setattr(bundle, "synthetic_transactions", synthetic)
    monkeypatch.setattr(bundle, "build", fake_build)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fraud_ml.bundle",
            "--demo",
            "--seed",
            "1",
            "--params",
            str(workspace / "params.yaml"),
            "--out",
            str(tmp_path / "bundle"),
        ],
    )

    bundle.main()

    assert observed == {"seed": 1, "rows": 100, "kind": "demo"}
