"""Integration tests for MLflow comparison, selection, gating, and packaging."""

from __future__ import annotations

import json
from pathlib import Path

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from ml_project_template.config import Settings
from ml_project_template.example_data import write_example_data
from ml_project_template.experiment import QualityGateError, run_experiment
from ml_project_template.predict import predict_record
from ml_project_template.prepare_data import prepare_data


def _experiment_paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "data_path": tmp_path / "data" / "example.csv",
        "artifact_path": tmp_path / "models" / "model.joblib",
        "metrics_path": tmp_path / "models" / "metrics.json",
        "model_card_path": tmp_path / "models" / "model-card.md",
    }


def _sqlite_tracking_uri(tmp_path: Path) -> str:
    return f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"


def test_experiment_compares_runs_and_packages_only_selected_model(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cv_folds=2, quality_gate_min_f1=0.0)
    paths = _experiment_paths(tmp_path)
    raw_path = tmp_path / "data" / "raw" / "example.csv"
    manifest_path = tmp_path / "data" / "processed" / "manifest.json"
    write_example_data(raw_path, rows=80, random_state=settings.random_state)
    prepared = prepare_data(
        input_path=raw_path,
        output_path=paths["data_path"],
        manifest_path=manifest_path,
        profile_path=tmp_path / "data" / "processed" / "profile.json",
        settings=settings,
    )
    tracking_uri = _sqlite_tracking_uri(tmp_path)
    experiment_name = "phase-3-success"

    result = run_experiment(
        **paths,
        settings=settings,
        tracking_uri=tracking_uri,
        experiment_name=experiment_name,
        quality_gate_min_f1=0.0,
        data_manifest_path=manifest_path,
    )

    assert result.artifact_path.is_file()
    assert result.metrics_path.is_file()
    assert result.model_card_path.is_file()
    assert set(result.candidate_run_ids) == {"baseline", "tuned"}
    assert len(set(result.candidate_run_ids.values())) == 2
    assert result.selected_candidate in result.candidate_run_ids
    assert result.selected_run_id == result.candidate_run_ids[result.selected_candidate]
    assert all(0 <= score <= 1 for score in result.selection_scores.values())
    assert result.model_uri

    metrics_payload = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics_payload["test_used_during_selection"] is False
    assert metrics_payload["quality_gate_passed"] is True
    assert len(metrics_payload["data_sha256"]) == 64
    assert metrics_payload["data_reference"] == prepared.output_reference

    experiment = mlflow.get_experiment_by_name(experiment_name)
    assert experiment is not None
    runs = MlflowClient().search_runs([experiment.experiment_id])
    assert len(runs) == 2
    selected_run = MlflowClient().get_run(result.selected_run_id)
    assert selected_run.data.tags["selection_status"] == "selected"
    assert selected_run.data.tags["quality_gate"] == "passed"
    assert "test_f1" in selected_run.data.metrics
    assert selected_run.data.params["data_reference"] == prepared.output_reference

    model_info = mlflow.models.get_model_info(result.model_uri)
    assert model_info.signature is not None
    assert model_info.metadata["selected_candidate"] == result.selected_candidate

    prediction = predict_record(
        {
            "numeric_feature_1": 0.25,
            "numeric_feature_2": -0.5,
            "category_feature_1": "unseen_category",
            "category_feature_2": "group_1",
        },
        result.artifact_path,
    )
    assert prediction["prediction"] in {0, 1}


def test_failed_quality_gate_does_not_package_model(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cv_folds=2)
    paths = _experiment_paths(tmp_path)
    write_example_data(paths["data_path"], rows=80, random_state=settings.random_state)

    with pytest.raises(QualityGateError, match="failed the packaging quality gate"):
        run_experiment(
            **paths,
            settings=settings,
            tracking_uri=_sqlite_tracking_uri(tmp_path),
            experiment_name="phase-3-failure",
            quality_gate_min_f1=1.0,
        )

    assert not paths["artifact_path"].exists()
    assert not paths["metrics_path"].exists()
    assert not paths["model_card_path"].exists()


def test_experiment_rejects_invalid_gate_before_tracking(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    paths = _experiment_paths(tmp_path)

    with pytest.raises(ValueError, match="quality_gate_min_f1"):
        run_experiment(
            **paths,
            settings=settings,
            tracking_uri=_sqlite_tracking_uri(tmp_path),
            experiment_name="invalid-gate",
            quality_gate_min_f1=1.1,
        )
