"""Manifest: the contract between the prepare stage and the training stage.

Prepare writes ``manifest.json`` next to its outputs. Training calls
:func:`verify` before reading anything, and refuses to run if the outputs
were changed after prepare or were produced with different parameters.
This keeps every reported result traceable to one data version, one set
of parameters, and one code commit.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_NAME = "manifest.json"
FORMAT_VERSION = 1


class ManifestMismatchError(RuntimeError):
    """Artifacts do not match their manifest or the expected parameters."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of a file, read in chunks to bound memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_version() -> dict[str, Any]:
    """Return the current git commit and whether the working tree is dirty.

    A dirty tree is recorded, not rejected, so experiments stay quick; the
    report should only cite results from a clean commit.
    """

    def git(*args: str) -> str:
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip() if result.returncode == 0 else ""

    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain"))}


def write(
    output_dir: Path,
    *,
    params_hash: str,
    inputs: list[Path],
    outputs: list[Path],
    details: dict[str, Any],
) -> Path:
    """Write the manifest for a finished prepare run.

    Args:
        output_dir: Directory holding the outputs; the manifest goes here.
        params_hash: Hash of the parameter sections that shaped the outputs.
        inputs: Raw files read by the stage.
        outputs: Files written by the stage, all inside ``output_dir``.
        details: Stage-specific facts, such as split boundaries and row counts.

    Returns:
        Path of the manifest.
    """
    manifest = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "code": code_version(),
        "params_hash": params_hash,
        "inputs": {p.name: sha256_file(p) for p in inputs},
        "outputs": {p.name: sha256_file(p) for p in outputs},
        **details,
    }
    path = output_dir / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def verify(output_dir: Path, *, expected_params_hash: str) -> dict[str, Any]:
    """Check that artifacts match their manifest and the current parameters.

    Args:
        output_dir: Directory written by prepare.
        expected_params_hash: Hash computed from the current ``params.yaml``.

    Returns:
        The manifest contents.

    Raises:
        ManifestMismatchError: If the manifest is missing, the parameters
            differ, or any output file is missing or changed.
    """
    path = output_dir / MANIFEST_NAME
    if not path.exists():
        raise ManifestMismatchError(f"no manifest in {output_dir}; run prepare first")
    manifest = json.loads(path.read_text())
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ManifestMismatchError("unsupported manifest format; rerun prepare")
    if manifest["params_hash"] != expected_params_hash:
        raise ManifestMismatchError("artifacts were built with different parameters; rerun prepare")
    for name, expected in manifest["outputs"].items():
        file = output_dir / name
        if not file.exists() or sha256_file(file) != expected:
            raise ManifestMismatchError(
                f"{name} is missing or changed since prepare; rerun prepare"
            )
    return manifest
