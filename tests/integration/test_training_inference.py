"""End-to-end proof that the training artifact accepts raw inference input."""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from ml_project_template.config import Settings
from ml_project_template.example_data import write_example_data
from ml_project_template.predict import predict_record
from ml_project_template.train import train_model


def test_training_artifact_is_compatible_with_raw_inference(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    data_path = tmp_path / "data" / "example.csv"
    artifact_path = tmp_path / "models" / "model.joblib"
    metrics_path = tmp_path / "models" / "metrics.json"
    write_example_data(data_path, rows=120, random_state=settings.random_state)

    _, metrics = train_model(data_path, artifact_path, metrics_path, settings)

    assert artifact_path.is_file()
    assert metrics_path.is_file()
    assert metrics["training_rows"] == 96
    assert metrics["test_rows"] == 24
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == metrics

    saved_model = joblib.load(artifact_path)
    assert list(saved_model.named_steps) == ["preprocessor", "estimator"]
    assert saved_model.feature_names_in_.tolist() == list(settings.feature_columns)

    raw_record = {
        "numeric_feature_1": 0.25,
        "numeric_feature_2": -0.5,
        "category_feature_1": "unseen_category",
        "category_feature_2": "group_1",
    }
    result = predict_record(raw_record, artifact_path)

    assert set(result) == {"prediction", "probability"}
    assert result["prediction"] in {0, 1}
    assert result["probability"] is not None
    assert 0.0 <= result["probability"] <= 1.0
