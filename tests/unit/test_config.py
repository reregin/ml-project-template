"""Tests for typed configuration invariants."""

from pathlib import Path

import pytest

from ml_project_template.config import Settings


def test_settings_resolve_authoritative_paths(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)

    assert settings.raw_data_path == tmp_path.resolve() / "data" / "raw" / "example.csv"
    assert settings.model_artifact_path == tmp_path.resolve() / "models" / "model.joblib"
    assert settings.metrics_path == tmp_path.resolve() / "models" / "metrics.json"
    assert settings.feature_columns == (
        "numeric_feature_1",
        "numeric_feature_2",
        "category_feature_1",
        "category_feature_2",
    )


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_settings_reject_invalid_test_size(tmp_path: Path, test_size: float) -> None:
    with pytest.raises(ValueError, match="test_size"):
        Settings(project_root=tmp_path, test_size=test_size)


def test_settings_reject_duplicate_features(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        Settings(
            project_root=tmp_path,
            numeric_features=("duplicate",),
            categorical_features=("duplicate",),
        )


def test_settings_reject_empty_target(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="target_column"):
        Settings(project_root=tmp_path, target_column=" ")


def test_settings_reject_empty_feature_set(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="At least one feature"):
        Settings(
            project_root=tmp_path,
            numeric_features=(),
            categorical_features=(),
        )


def test_settings_reads_project_root_environment_variable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ML_PROJECT_ROOT", str(tmp_path))

    settings = Settings()

    assert settings.project_root == tmp_path.resolve()


@pytest.mark.parametrize("cv_folds", [0, 1])
def test_settings_reject_invalid_cv_folds(tmp_path: Path, cv_folds: int) -> None:
    with pytest.raises(ValueError, match="cv_folds"):
        Settings(project_root=tmp_path, cv_folds=cv_folds)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_settings_reject_invalid_quality_gate(tmp_path: Path, threshold: float) -> None:
    with pytest.raises(ValueError, match="quality_gate_min_f1"):
        Settings(project_root=tmp_path, quality_gate_min_f1=threshold)


def test_settings_reject_unsupported_selection_metric(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="selection_metric='f1'"):
        Settings(project_root=tmp_path, selection_metric="accuracy")


def test_settings_resolve_tracking_and_model_card_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    settings = Settings(project_root=tmp_path)

    assert settings.model_card_path == tmp_path.resolve() / "models" / "model-card.md"
    assert settings.tracking_uri.startswith("sqlite:///")
    assert settings.tracking_uri.endswith("/mlflow.db")

    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://tracking.example.invalid")
    assert Settings(project_root=tmp_path).tracking_uri == "https://tracking.example.invalid"
