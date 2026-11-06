from pathlib import Path

import pytest

from fraud_ml.config import SplitParams, load_params


def test_paths_resolve_relative_to_params_file(workspace: Path):
    params = load_params(workspace / "params.yaml")
    assert params.data.raw_dir == (workspace / "data" / "raw").resolve()


@pytest.mark.parametrize("train,valid", [(0.0, 0.1), (0.9, 0.1), (0.5, 1.2)])
def test_invalid_fractions_rejected(train, valid):
    with pytest.raises(ValueError):
        SplitParams(train_fraction=train, valid_fraction=valid)


def test_hash_ignores_paths_but_tracks_values(workspace: Path, tmp_path_factory):
    original = (workspace / "params.yaml").read_text()
    moved = tmp_path_factory.mktemp("elsewhere") / "params.yaml"
    moved.write_text(original.replace("data/raw", "other/raw"))
    changed = tmp_path_factory.mktemp("changed") / "params.yaml"
    changed.write_text(original.replace("0.7", "0.6"))

    base = load_params(workspace / "params.yaml").stage_hash("split", "preprocess")
    assert load_params(moved).stage_hash("split", "preprocess") == base
    assert load_params(changed).stage_hash("split", "preprocess") != base
