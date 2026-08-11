"""MLflow setup and reproducibility artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mlflow
import pandas as pd

from ml_project_template.config import Settings
from ml_project_template.data_contract import build_data_profile
from ml_project_template.lineage import git_revision as git_revision
from ml_project_template.lineage import sha256_file as sha256_file


def configure_tracking(tracking_uri: str, experiment_name: str) -> str:
    """Configure MLflow explicitly and return the resolved experiment ID."""

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.set_experiment(experiment_name)
    return str(experiment.experiment_id)


def serializable_settings(settings: Settings) -> dict[str, Any]:
    """Convert settings to JSON-safe values for run reproducibility."""

    values = asdict(settings)
    values["project_root"] = str(settings.project_root)
    values["feature_columns"] = list(settings.feature_columns)
    return values


def dataset_profile(data: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Create the shared profile plus convenient aggregate run metadata."""

    profile = build_data_profile(data, settings)
    profile.update(
        {
            "missing_values": int(data.isna().sum().sum()),
            "feature_columns": list(settings.feature_columns),
            "target_column": settings.target_column,
        }
    )
    return profile


def log_reproducibility_artifacts(
    *,
    settings: Settings,
    data: pd.DataFrame,
    data_sha256: str,
    code_revision: str,
    data_lineage: dict[str, Any] | None = None,
    extra_artifacts: dict[str, Any] | None = None,
) -> None:
    """Log configuration, data profile, lineage, and candidate diagnostics."""

    with TemporaryDirectory() as temporary_directory:
        artifact_directory = Path(temporary_directory)
        lineage = {
            "data_sha256": data_sha256,
            "code_revision": code_revision,
        }
        if data_lineage:
            lineage.update(data_lineage)
        payloads: dict[str, Any] = {
            "resolved_config.json": serializable_settings(settings),
            "dataset_profile.json": dataset_profile(data, settings),
            "lineage.json": lineage,
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
