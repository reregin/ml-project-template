"""MLflow setup and reproducibility artifact helpers."""

from __future__ import annotations

import hashlib
import json
import subprocess  # nosec B404
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import pandas as pd

from ml_project_template.config import Settings


def configure_tracking(tracking_uri: str, experiment_name: str) -> str:
    """Configure MLflow explicitly and return the resolved experiment ID."""

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return str(experiment.experiment_id)


def sha256_file(path: Path) -> str:
    """Return a stable hash for the exact training-data bytes."""

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


def serializable_settings(settings: Settings) -> dict[str, Any]:
    """Convert settings to JSON-safe values for run reproducibility."""

    values = asdict(settings)
    values["project_root"] = str(settings.project_root)
    values["feature_columns"] = list(settings.feature_columns)
    return values


def dataset_profile(data: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Create a small non-sensitive profile for the tracked fixture."""

    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "missing_values": int(data.isna().sum().sum()),
        "feature_columns": list(settings.feature_columns),
        "target_column": settings.target_column,
        "target_distribution": {
            str(label): int(count)
            for label, count in data[settings.target_column].value_counts().sort_index().items()
        },
    }


def log_reproducibility_artifacts(
    *,
    settings: Settings,
    data: pd.DataFrame,
    data_sha256: str,
    code_revision: str,
    extra_artifacts: dict[str, Any] | None = None,
) -> None:
    """Log configuration, data profile, lineage, and candidate diagnostics."""

    with TemporaryDirectory() as temporary_directory:
        artifact_directory = Path(temporary_directory)
        payloads: dict[str, Any] = {
            "resolved_config.json": serializable_settings(settings),
            "dataset_profile.json": dataset_profile(data, settings),
            "lineage.json": {
                "data_sha256": data_sha256,
                "code_revision": code_revision,
            },
        }
        if extra_artifacts:
            payloads.update(extra_artifacts)

        for filename, payload in payloads.items():
            destination = artifact_directory / filename
            if isinstance(payload, pd.DataFrame):
                payload.to_csv(destination, index=False)
            else:
                destination.write_text(
                    json.dumps(payload, indent=2, sort_keys=True),
                    encoding="utf-8",
                )

        mlflow.log_artifacts(str(artifact_directory), artifact_path="evidence")
