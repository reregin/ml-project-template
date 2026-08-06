"""Tests for artifact loading, input validation, and response semantics."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pytest

from ml_project_template.predict import load_model, predict_record, validate_record


class ZeroProbabilityModel:
    """Small serializable test double proving 0.0 is not treated as missing."""

    feature_names_in_ = np.array(["feature"])
    classes_ = np.array([0, 1])

    def predict(self, frame: Any) -> np.ndarray:
        return np.array([0])

    def predict_proba(self, frame: Any) -> np.ndarray:
        return np.array([[1.0, 0.0]])


class PredictionOnlyModel:
    """Test double for estimators without probability support."""

    feature_names_in_ = np.array(["feature"])

    def predict(self, frame: Any) -> np.ndarray:
        return np.array([1])


def test_load_model_rejects_missing_artifact(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Run training first"):
        load_model(tmp_path / "missing.joblib")


def test_validate_record_reports_all_missing_columns() -> None:
    with pytest.raises(ValueError, match="feature_b.*feature_c"):
        validate_record({"feature_a": 1}, ["feature_a", "feature_b", "feature_c"])


def test_predict_record_preserves_zero_probability(tmp_path: Path) -> None:
    artifact_path = tmp_path / "zero-probability.joblib"
    joblib.dump(ZeroProbabilityModel(), artifact_path)

    result = predict_record({"feature": 1.0}, artifact_path)

    assert result == {"prediction": 0, "probability": 0.0}
    assert set(result) == {"prediction", "probability"}


def test_predict_record_supports_estimator_without_probability(tmp_path: Path) -> None:
    artifact_path = tmp_path / "prediction-only.joblib"
    joblib.dump(PredictionOnlyModel(), artifact_path)

    result = predict_record({"feature": 1.0}, artifact_path)

    assert result == {"prediction": 1, "probability": None}
