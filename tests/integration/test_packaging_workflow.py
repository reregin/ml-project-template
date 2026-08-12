"""End-to-end proof of the immutable Phase 5 model-package contract."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import mlflow.pyfunc
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml_project_template.config import Settings
from ml_project_template.example_data import write_example_data
from ml_project_template.model_package import (
    ModelPackageError,
    build_model_package,
    validate_model_package,
)
from ml_project_template.prepare_data import prepare_data
from ml_project_template.service import create_app


def test_build_validate_predict_and_detect_package_tampering(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, cv_folds=2, quality_gate_min_f1=0.0)
    raw_path = tmp_path / "data" / "raw" / "example.csv"
    data_path = tmp_path / "data" / "processed" / "training.csv"
    manifest_path = tmp_path / "data" / "processed" / "manifest.json"
    profile_path = tmp_path / "data" / "processed" / "profile.json"
    request_path = tmp_path / "request.json"
    request = {
        "numeric_feature_1": 0.25,
        "numeric_feature_2": -0.5,
        "category_feature_1": "unseen_category",
        "category_feature_2": "group_1",
    }
    request_path.write_text(json.dumps(request), encoding="utf-8")
    write_example_data(raw_path, rows=80, random_state=settings.random_state)
    prepared = prepare_data(
        input_path=raw_path,
        output_path=data_path,
        manifest_path=manifest_path,
        profile_path=profile_path,
        settings=settings,
    )

    result = build_model_package(
        data_path=data_path,
        data_manifest_path=manifest_path,
        data_profile_path=profile_path,
        request_path=request_path,
        output_root=tmp_path / "packages",
        settings=settings,
        tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        experiment_name="phase-5-package",
        quality_gate_min_f1=0.0,
        artifact_version="abc1234",
    )

    package_manifest = validate_model_package(result.package_path)
    result_payload = result.as_dict()
    assert result.package_path.name == "abc1234"
    assert result_payload["artifact_version"] == "abc1234"
    assert result_payload["package_path"] == str(result.package_path)
    assert result.package_manifest_path.is_file()
    assert package_manifest["artifact_version"] == "abc1234"
    assert package_manifest["data_reference"] == prepared.output_reference
    assert package_manifest["selected_run_id"] == result.selected_run_id
    assert result_payload["model_uri"] == result.model_uri
    assert (result.package_path / "model" / "MLmodel").is_file()
    assert (result.package_path / "model" / "requirements.txt").is_file()

    model = mlflow.pyfunc.load_model(str(result.package_path / "model"))
    prediction = model.predict(pd.DataFrame([request]))
    assert int(prediction[0]) in {0, 1}

    mounted_package_path = tmp_path / "mounted-model-package"
    shutil.copytree(result.package_path, mounted_package_path)
    mounted_manifest = validate_model_package(mounted_package_path)
    assert mounted_manifest["artifact_version"] == "abc1234"

    with TestClient(create_app(mounted_package_path)) as client:
        api_prediction = client.post("/predict", json=request)
        service_metrics = client.get("/metrics")
    assert api_prediction.status_code == 200
    assert api_prediction.json()["prediction"] == int(prediction[0])
    assert api_prediction.json()["model"]["artifact_version"] == "abc1234"
    assert "ml_model_ready 1.0" in service_metrics.text
    assert "ml_prediction_requests_total 1.0" in service_metrics.text

    with pytest.raises(FileExistsError, match="Immutable package already exists"):
        build_model_package(
            data_path=data_path,
            data_manifest_path=manifest_path,
            data_profile_path=profile_path,
            request_path=request_path,
            output_root=tmp_path / "packages",
            settings=settings,
            tracking_uri="unused",
            experiment_name="unused",
            quality_gate_min_f1=0.0,
            artifact_version="abc1234",
        )

    card_path = result.package_path / "model-card.md"
    card_path.write_text(card_path.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
    with pytest.raises(ModelPackageError, match="integrity validation"):
        validate_model_package(result.package_path)
