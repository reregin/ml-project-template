"""HTTP contract tests for liveness, readiness, prediction, and stable errors."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from ml_project_template.service import (
    DEFAULT_MAX_REQUEST_BYTES,
    ModelMetadata,
    PredictionRequest,
    ServiceRuntime,
    _default_package_path,
    _default_request_limit,
    create_app,
    parse_args,
)


class FakeModel:
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series([1] * len(frame))


class FailingModel:
    def predict(self, frame: pd.DataFrame) -> pd.Series:
        raise RuntimeError("sensitive internal failure")


class ReadyRuntime(ServiceRuntime):
    def __init__(self, model: Any | None = None) -> None:
        super().__init__(Path("unused"))
        self.model = model or FakeModel()
        self.metadata = ModelMetadata(
            artifact_version="abc123",
            selected_run_id="run-123",
            data_reference="sha256:data",
        )

    def load(self) -> None:
        return


class UnavailableRuntime(ServiceRuntime):
    def __init__(self) -> None:
        super().__init__(Path("missing"))

    def load(self) -> None:
        self.load_error = "ModelPackageError"


VALID_REQUEST = {
    "numeric_feature_1": 0.25,
    "numeric_feature_2": -0.5,
    "category_feature_1": "unseen_category",
    "category_feature_2": "group_1",
}


def test_live_ready_and_prediction_include_model_identity() -> None:
    with TestClient(create_app(runtime=ReadyRuntime())) as client:
        live = client.get("/live")
        ready = client.get("/ready")
        prediction = client.post("/predict", json=VALID_REQUEST)

    assert live.status_code == 200
    assert live.json() == {"status": "live", "api_version": "1"}
    assert ready.status_code == 200
    assert ready.json()["artifact_version"] == "abc123"
    assert prediction.status_code == 200
    assert prediction.json() == {
        "prediction": 1,
        "model": {
            "artifact_version": "abc123",
            "selected_run_id": "run-123",
            "data_reference": "sha256:data",
        },
    }


def test_metrics_distinguish_success_invalid_and_model_state() -> None:
    app = create_app(runtime=ReadyRuntime(), max_request_bytes=128)
    with TestClient(app) as client:
        assert client.post("/predict", json=VALID_REQUEST).status_code == 200
        assert client.post("/predict", json={"unexpected": "field"}).status_code == 422
        assert (
            client.post(
                "/predict",
                content=b"x" * 129,
                headers={"content-type": "application/json"},
            ).status_code
            == 413
        )
        metrics = client.get("/metrics")

    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert "ml_model_ready 1.0" in metrics.text
    assert "ml_prediction_requests_total 3.0" in metrics.text
    assert 'ml_prediction_responses_total{outcome="success"} 1.0' in metrics.text
    assert 'ml_prediction_responses_total{outcome="invalid"} 2.0' in metrics.text
    assert "ml_prediction_duration_seconds_count 3.0" in metrics.text
    assert "ml_prediction_rows_total 1.0" in metrics.text
    assert 'ml_prediction_outputs_total{class="1"} 1.0' in metrics.text
    assert "ml_prediction_requests_in_progress 0.0" in metrics.text


def test_metrics_show_model_unavailable_without_exposing_load_error() -> None:
    with TestClient(create_app(runtime=UnavailableRuntime())) as client:
        assert client.post("/predict", json=VALID_REQUEST).status_code == 503
        metrics = client.get("/metrics")

    assert "ml_model_ready 0.0" in metrics.text
    assert 'ml_prediction_responses_total{outcome="unavailable"} 1.0' in metrics.text
    assert "ModelPackageError" not in metrics.text


def test_invalid_and_oversized_requests_have_stable_4xx_errors() -> None:
    app = create_app(runtime=ReadyRuntime(), max_request_bytes=128)
    with TestClient(app) as client:
        invalid = client.post("/predict", json={"unexpected": "field"})
        oversized = client.post(
            "/predict",
            content=b"x" * 129,
            headers={"content-type": "application/json"},
        )

    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "invalid_request"
    assert {detail["field"] for detail in invalid.json()["error"]["details"]} >= {
        "numeric_feature_1",
        "unexpected",
    }
    assert oversized.status_code == 413
    assert oversized.json() == {
        "error": {
            "code": "request_too_large",
            "message": "Request body is too large.",
        }
    }


def test_liveness_survives_model_down_while_readiness_and_prediction_fail() -> None:
    with TestClient(create_app(runtime=UnavailableRuntime())) as client:
        assert client.get("/live").status_code == 200
        ready = client.get("/ready")
        prediction = client.post("/predict", json=VALID_REQUEST)

    assert ready.status_code == 503
    assert prediction.status_code == 503
    assert ready.json()["error"]["code"] == "model_unavailable"
    assert prediction.json() == ready.json()


def test_unexpected_prediction_failure_hides_internal_details() -> None:
    app = create_app(runtime=ReadyRuntime(FailingModel()))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/predict", json=VALID_REQUEST)

    assert response.status_code == 500
    assert response.json()["error"] == {
        "code": "internal_error",
        "message": "Prediction could not be completed.",
    }
    assert "sensitive" not in response.text


def test_runtime_load_failure_and_direct_prediction_guard(tmp_path: Path) -> None:
    runtime = ServiceRuntime(tmp_path / "missing-package")

    runtime.load()

    assert runtime.ready is False
    assert runtime.load_error == "ModelPackageError"
    request = PredictionRequest(**VALID_REQUEST)
    with pytest.raises(RuntimeError, match="Model is not ready"):
        runtime.predict(request)


def test_environment_defaults_and_cli_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODEL_PACKAGE_PATH", str(tmp_path / "package"))
    monkeypatch.setenv("MAX_REQUEST_BYTES", "invalid")
    assert _default_package_path() == tmp_path / "package"
    assert _default_request_limit() == DEFAULT_MAX_REQUEST_BYTES

    monkeypatch.setenv("MAX_REQUEST_BYTES", "-1")
    assert _default_request_limit() == DEFAULT_MAX_REQUEST_BYTES
    monkeypatch.setenv("MAX_REQUEST_BYTES", "2048")
    assert _default_request_limit() == 2048

    monkeypatch.setattr(
        sys,
        "argv",
        ["serve", "--package", str(tmp_path), "--host", "127.0.0.1", "--port", "9000"],
    )
    args = parse_args()
    assert args.package == tmp_path
    assert args.host == "127.0.0.1"
    assert args.port == 9000
