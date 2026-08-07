"""Tests for reproducibility hashes, profiles, settings, and revision lookup."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from ml_project_template.config import Settings
from ml_project_template.tracking import (
    dataset_profile,
    git_revision,
    serializable_settings,
    sha256_file,
)


def test_sha256_file_tracks_exact_bytes(tmp_path: Path) -> None:
    path = tmp_path / "data.bin"
    path.write_bytes(b"phase-three")

    assert sha256_file(path) == hashlib.sha256(b"phase-three").hexdigest()


def test_serializable_settings_and_dataset_profile(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    config_payload = serializable_settings(settings)
    profile = dataset_profile(example_data, settings)

    assert config_payload["project_root"] == str(settings.project_root)
    assert config_payload["feature_columns"] == list(settings.feature_columns)
    assert profile["rows"] == 120
    assert profile["columns"] == 5
    assert profile["target_distribution"] == {"0": 60, "1": 60}
    assert profile["missing_values"] > 0


def test_git_revision_returns_unavailable_outside_repository(tmp_path: Path) -> None:
    assert git_revision(tmp_path) == "unavailable"
