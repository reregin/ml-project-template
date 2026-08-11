"""Focused tests for package versions and early build failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import ml_project_template.model_package as package_module
from ml_project_template.config import Settings
from ml_project_template.model_package import (
    ModelPackageError,
    build_model_package,
    resolve_artifact_version,
    validate_model_package,
)


def test_resolve_artifact_version_prefers_explicit_then_ci(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.setenv("GITHUB_SHA", "ci123")

    assert resolve_artifact_version(settings, "release-1") == "release-1"
    assert resolve_artifact_version(settings, None) == "ci123"


def test_resolve_artifact_version_rejects_unsafe_path(tmp_path: Path) -> None:
    with pytest.raises(ModelPackageError, match="Artifact version"):
        resolve_artifact_version(Settings(project_root=tmp_path), "../overwrite")


def test_resolve_artifact_version_uses_local_outside_git(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)

    assert resolve_artifact_version(Settings(project_root=tmp_path), None) == "local"


def test_default_version_rejects_dirty_worktree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(project_root=tmp_path)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(package_module, "git_revision", lambda _: "abc123")
    monkeypatch.setattr(package_module, "git_worktree_is_dirty", lambda _: True)

    with pytest.raises(ModelPackageError, match="uncommitted changes"):
        resolve_artifact_version(settings, None)

    assert resolve_artifact_version(settings, "dirty-local") == "dirty-local"


def test_validate_model_package_rejects_missing_and_invalid_manifest(tmp_path: Path) -> None:
    package_path = tmp_path / "version"
    package_path.mkdir()
    with pytest.raises(ModelPackageError, match="not found"):
        validate_model_package(package_path)

    (package_path / "package-manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ModelPackageError, match="invalid"):
        validate_model_package(package_path)

    (package_path / "package-manifest.json").write_text(
        json.dumps({"manifest_version": 999, "artifact_version": "version", "files": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ModelPackageError, match="Unsupported"):
        validate_model_package(package_path)


def test_failed_training_never_publishes_partial_package(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(project_root=tmp_path)
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(dict.fromkeys(settings.feature_columns, "value")),
        encoding="utf-8",
    )
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("{}", encoding="utf-8")
    manifest_path = tmp_path / "data-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input": {"immutable_reference": "sha256:raw"},
                "output": {
                    "sha256": "prepared",
                    "immutable_reference": "sha256:prepared",
                },
            }
        ),
        encoding="utf-8",
    )

    def fail_experiment(**_: object) -> None:
        raise RuntimeError("training failed")

    monkeypatch.setattr(package_module, "run_experiment", fail_experiment)
    output_root = tmp_path / "packages"
    with pytest.raises(RuntimeError, match="training failed"):
        build_model_package(
            data_path=tmp_path / "training.csv",
            data_manifest_path=manifest_path,
            data_profile_path=profile_path,
            request_path=request_path,
            output_root=output_root,
            settings=settings,
            tracking_uri="unused",
            experiment_name="unused",
            quality_gate_min_f1=0.7,
            artifact_version="failed-build",
        )

    assert not (output_root / "failed-build").exists()
