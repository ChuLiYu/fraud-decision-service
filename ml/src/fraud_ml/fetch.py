"""Stage 1: download the IEEE-CIS competition files with the Kaggle CLI.

Prerequisites (done once by the owner, never by code):

* A Kaggle account that has accepted the competition rules.
* A Kaggle API token configured locally; it is never committed.

Only the two labeled training tables are kept. The competition test tables
have no public labels, so this project does not use them (dataset card).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from fraud_ml.config import load_params

COMPETITION = "ieee-fraud-detection"
REQUIRED_FILES = ("train_transaction.csv", "train_identity.csv")


def fetch(raw_dir: Path) -> list[Path]:
    """Download and unzip the labeled tables into ``raw_dir``.

    Skips the download when both files already exist, so the stage is safe
    to re-run.

    Args:
        raw_dir: Destination directory, ignored by git.

    Returns:
        Paths of the required CSV files.

    Raises:
        RuntimeError: If the Kaggle CLI is missing or the download fails.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    targets = [raw_dir / name for name in REQUIRED_FILES]
    if all(path.exists() for path in targets):
        return targets

    if shutil.which("kaggle") is None:
        raise RuntimeError("Kaggle CLI not found; run `uv sync --extra fetch` first")
    for name in REQUIRED_FILES:
        result = subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", name, "-p", raw_dir],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Kaggle download of {name} failed; check that the competition rules are "
                f"accepted and the API token is configured.\n{result.stderr.strip()}"
            )
        archive = raw_dir / f"{name}.zip"
        if archive.exists():  # the CLI zips single files larger than a threshold
            with zipfile.ZipFile(archive) as zf:
                zf.extract(name, raw_dir)
            archive.unlink()

    missing = [path.name for path in targets if not path.exists()]
    if missing:
        raise RuntimeError(f"Download finished but files are missing: {missing}")
    return targets


def main() -> None:
    """Command-line entry point: ``python -m fraud_ml.fetch``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--params", type=Path, default=Path("params.yaml"))
    args = parser.parse_args()
    for path in fetch(load_params(args.params).data.raw_dir):
        print(f"ready: {path}")


if __name__ == "__main__":
    main()
