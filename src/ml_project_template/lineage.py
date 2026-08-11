"""Content-addressed data-lineage helpers shared by preparation and training."""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404
from pathlib import Path
from typing import Any, cast


class DataLineageError(ValueError):
    """Raised when a prepared dataset no longer matches its manifest."""


def sha256_file(path: Path) -> str:
    """Return a stable hash for the exact file bytes."""

    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_revision(project_root: Path) -> str:
    """Return the current Git commit when available."""

    result = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        capture_output=True,
        check=False,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def read_data_manifest(path: Path) -> dict[str, Any]:
    """Read a preparation manifest and validate its minimum structure."""

    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise DataLineageError(f"Data manifest not found at {resolved_path}.")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
        expected_hash = payload["output"]["sha256"]
        output_reference = payload["output"]["immutable_reference"]
        input_reference = payload["input"]["immutable_reference"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise DataLineageError(f"Invalid data manifest at {resolved_path}.") from error
    if (
        not isinstance(expected_hash, str)
        or output_reference != f"sha256:{expected_hash}"
        or not isinstance(input_reference, str)
        or not input_reference.startswith("sha256:")
    ):
        raise DataLineageError(f"Invalid data manifest at {resolved_path}.")
    return cast(dict[str, Any], payload)


def verify_data_manifest(data_path: Path, manifest_path: Path) -> dict[str, Any]:
    """Verify prepared bytes against the recorded content hash."""

    resolved_data_path = data_path.expanduser().resolve()
    if not resolved_data_path.is_file():
        raise DataLineageError(f"Prepared data not found at {resolved_data_path}.")
    manifest = read_data_manifest(manifest_path)
    expected_hash = str(manifest["output"]["sha256"])
    actual_hash = sha256_file(resolved_data_path)
    if actual_hash != expected_hash:
        raise DataLineageError(
            "Prepared data hash does not match its manifest: "
            f"expected {expected_hash}, received {actual_hash}."
        )
    return manifest
