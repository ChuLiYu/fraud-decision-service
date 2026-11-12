"""Stage 2: turn raw tables into the artifacts that training reads.

Steps: load and validate, add time-safe features, split by time, fit
preprocessing on the training range only, transform all three ranges, and write:

* ``train.parquet``, ``valid.parquet``, ``test.parquet``: transformed rows.
* ``preprocessor.json``: the fitted preprocessing, reused by serving later.
* ``manifest.json``: hashes, parameters, split boundaries, and code version.

Training reads only these files and calls :func:`fraud_ml.manifest.verify`
first, so it can never see raw data or mismatched artifacts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from fraud_ml import manifest
from fraud_ml.config import Params, load_params
from fraud_ml.data import LABEL, load_raw, validate
from fraud_ml.features import add_features
from fraud_ml.fetch import REQUIRED_FILES
from fraud_ml.preprocess import Preprocessor
from fraud_ml.split import time_split

PARAM_SECTIONS = ("features", "split", "preprocess")


def params_hash(params: Params) -> str:
    """Hash of the parameter sections that shape prepare's outputs."""
    return params.stage_hash(*PARAM_SECTIONS)


def prepare(params: Params) -> Path:
    """Run the prepare stage end to end.

    Args:
        params: Loaded pipeline parameters.

    Returns:
        Path of the written manifest.
    """
    raw_dir, out_dir = params.data.raw_dir, params.data.processed_dir
    df = load_raw(raw_dir)
    validate(df)
    # Past-only features are computed before the split; see fraud_ml.features.
    df = add_features(df, params.features)
    split = time_split(df, params.split)

    preprocessor = Preprocessor(params.preprocess.min_category_count).fit(split.train)

    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    ranges = {}
    for name, frame in (("train", split.train), ("valid", split.valid), ("test", split.test)):
        path = out_dir / f"{name}.parquet"
        preprocessor.transform(frame).to_parquet(path, index=False)
        outputs.append(path)
        ranges[name] = {"rows": len(frame), "fraud_rate": round(float(frame[LABEL].mean()), 6)}
    preprocessor_path = out_dir / "preprocessor.json"
    preprocessor.save(preprocessor_path)
    outputs.append(preprocessor_path)

    return manifest.write(
        out_dir,
        params_hash=params_hash(params),
        inputs=[raw_dir / name for name in REQUIRED_FILES],
        outputs=outputs,
        details={
            "split": {
                "valid_start": split.valid_start,
                "test_start": split.test_start,
                "ranges": ranges,
            },
            "features": {
                "numeric": len(preprocessor.numeric),
                "categorical": len(preprocessor.categorical),
            },
        },
    )


def main() -> None:
    """Command-line entry point: ``python -m fraud_ml.prepare``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    print(f"wrote {prepare(load_params(args.params))}")


if __name__ == "__main__":
    main()
