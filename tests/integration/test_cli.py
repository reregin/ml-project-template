"""Command-level tests for the documented Phase 1 interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from ml_project_template import example_data, predict, prepare_data, train


class ZeroProbabilityModel:
    feature_names_in_ = np.array(["feature"])
    classes_ = np.array([0, 1])

    def predict(self, frame: Any) -> np.ndarray:
        return np.array([0])

    def predict_proba(self, frame: Any) -> np.ndarray:
        return np.array([[1.0, 0.0]])


def test_generate_train_and_predict_cli_with_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_data_path = tmp_path / "raw.csv"
    data_path = tmp_path / "training.csv"
    manifest_path = tmp_path / "manifest.json"
    artifact_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(
            {
                "numeric_feature_1": 0.25,
                "numeric_feature_2": -0.5,
                "category_feature_1": "unseen_category",
                "category_feature_2": "group_1",
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["example-data", "--output", str(raw_data_path), "--rows", "40"],
    )
    example_data.main()
    assert raw_data_path.is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare-data",
            "--input",
            str(raw_data_path),
            "--output",
            str(data_path),
            "--manifest",
            str(manifest_path),
            "--profile",
            str(tmp_path / "profile.json"),
        ],
    )
    prepare_data.main()
    preparation_output = json.loads(capsys.readouterr().out)
    assert preparation_output["output_reference"].startswith("sha256:")

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--data",
            str(data_path),
            "--data-manifest",
            str(manifest_path),
            "--artifact",
            str(artifact_path),
            "--metrics",
            str(metrics_path),
        ],
    )
    train.main()
    training_output = json.loads(capsys.readouterr().out)
    assert training_output["training_rows"] == 32
    assert training_output["data_reference"] == preparation_output["output_reference"]
    assert artifact_path.is_file()
    assert metrics_path.is_file()

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--artifact",
            str(artifact_path),
            "--input-file",
            str(request_path),
        ],
    )
    predict.main()
    prediction_output = json.loads(capsys.readouterr().out)
    assert set(prediction_output) == {"prediction", "probability"}


def test_predict_cli_accepts_inline_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    artifact_path = tmp_path / "model.joblib"
    joblib.dump(ZeroProbabilityModel(), artifact_path)
    payload = json.dumps({"feature": 1.0})
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "predict",
            "--artifact",
            str(artifact_path),
            "--input-json",
            payload,
        ],
    )

    predict.main()

    assert json.loads(capsys.readouterr().out) == {
        "prediction": 0,
        "probability": 0.0,
    }


def test_predict_cli_rejects_non_object_json(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["predict", "--input-json", "[]"])

    with pytest.raises(ValueError, match="JSON object"):
        predict.main()
