"""Typed FastAPI service for an exact validated Phase 5 model package."""

from __future__ import annotations

import argparse
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from ml_project_template.config import configure_logging
from ml_project_template.model_package import validate_model_package

LOGGER = logging.getLogger(__name__)
API_VERSION = "1"
DEFAULT_MAX_REQUEST_BYTES = 16 * 1024


class PredictionRequest(BaseModel):
    """Raw fields accepted by the generic binary-classification package."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    numeric_feature_1: float = Field(allow_inf_nan=False)
    numeric_feature_2: float = Field(allow_inf_nan=False)
    category_feature_1: str = Field(min_length=1, max_length=256)
    category_feature_2: str = Field(min_length=1, max_length=256)


class ModelMetadata(BaseModel):
    artifact_version: str
    selected_run_id: str
    data_reference: str


class PredictionResponse(BaseModel):
    prediction: int
    model: ModelMetadata


class ServiceRuntime:
    """Own model readiness and immutable package metadata for one process."""

    def __init__(self, package_path: Path) -> None:
        self.package_path = package_path.expanduser().resolve()
        self.model: Any | None = None
        self.metadata: ModelMetadata | None = None
        self.load_error: str | None = None

    def load(self) -> None:
        """Validate the complete package before making the model ready."""

        self.model = None
        self.metadata = None
        self.load_error = None
        try:
            manifest = validate_model_package(self.package_path)
            self.model = mlflow.pyfunc.load_model(str(self.package_path / "model"))
            self.metadata = ModelMetadata(
                artifact_version=str(manifest["artifact_version"]),
                selected_run_id=str(manifest["selected_run_id"]),
                data_reference=str(manifest["data_reference"]),
            )
        except Exception as error:  # readiness must represent any load failure
            self.load_error = type(error).__name__
            LOGGER.exception(
                "event=model_load_failed package=%s error_type=%s",
                self.package_path,
                self.load_error,
            )

    @property
    def ready(self) -> bool:
        return self.model is not None and self.metadata is not None

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        model = self.model
        metadata = self.metadata
        if model is None or metadata is None:
            raise RuntimeError("Model is not ready.")
        frame = pd.DataFrame([request.model_dump()])
        values = model.predict(frame)
        prediction = int(values.iloc[0] if hasattr(values, "iloc") else values[0])
        return PredictionResponse(prediction=prediction, model=metadata)


def _error_response(status_code: int, code: str, message: str, **extra: Any) -> JSONResponse:
    payload: dict[str, Any] = {
        "error": {
            "code": code,
            "message": message,
        }
    }
    payload["error"].update(extra)
    return JSONResponse(status_code=status_code, content=payload)


def _default_package_path() -> Path:
    configured_path = os.getenv("MODEL_PACKAGE_PATH", "artifacts/model-packages/current")
    return Path(configured_path)


def _default_request_limit() -> int:
    configured_limit = os.getenv("MAX_REQUEST_BYTES")
    if configured_limit is None:
        return DEFAULT_MAX_REQUEST_BYTES
    try:
        limit = int(configured_limit)
    except ValueError:
        return DEFAULT_MAX_REQUEST_BYTES
    return limit if limit > 0 else DEFAULT_MAX_REQUEST_BYTES


def create_app(
    package_path: Path | None = None,
    *,
    runtime: ServiceRuntime | None = None,
    max_request_bytes: int | None = None,
) -> FastAPI:
    """Create an isolated application for production startup or tests."""

    service_runtime = runtime or ServiceRuntime(package_path or _default_package_path())
    request_limit = max_request_bytes or _default_request_limit()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        service_runtime.load()
        yield

    application = FastAPI(
        title="ML Project Template Prediction API",
        version=API_VERSION,
        lifespan=lifespan,
    )

    @application.middleware("http")
    async def enforce_request_size(request: Request, call_next: Any) -> Any:
        if request.url.path == "/predict":
            content_length = request.headers.get("content-length")
            if content_length and content_length.isdigit() and int(content_length) > request_limit:
                return _error_response(413, "request_too_large", "Request body is too large.")
            body = await request.body()
            if len(body) > request_limit:
                return _error_response(413, "request_too_large", "Request body is too large.")
        return await call_next(request)

    @application.exception_handler(RequestValidationError)
    async def validation_error_handler(_: Request, error: RequestValidationError) -> JSONResponse:
        details = [
            {
                "field": ".".join(str(part) for part in item["loc"] if part != "body"),
                "type": item["type"],
            }
            for item in error.errors()
        ]
        return _error_response(
            422,
            "invalid_request",
            "Request validation failed.",
            details=details,
        )

    @application.exception_handler(Exception)
    async def unexpected_error_handler(_: Request, error: Exception) -> JSONResponse:
        LOGGER.exception("event=request_failed error_type=%s", type(error).__name__)
        return _error_response(500, "internal_error", "Prediction could not be completed.")

    @application.get("/live")
    async def live() -> dict[str, str]:
        return {"status": "live", "api_version": API_VERSION}

    @application.get("/ready", response_model=ModelMetadata)
    async def ready() -> Any:
        if not service_runtime.ready:
            return _error_response(503, "model_unavailable", "Model is not ready.")
        return service_runtime.metadata

    @application.post("/predict", response_model=PredictionResponse)
    async def predict(request: PredictionRequest) -> PredictionResponse | JSONResponse:
        if not service_runtime.ready:
            return _error_response(503, "model_unavailable", "Model is not ready.")
        return service_runtime.predict(request)

    application.state.runtime = service_runtime
    application.state.max_request_bytes = request_limit
    return application


app = create_app()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, default=_default_package_path())
    parser.add_argument("--host", default=os.getenv("API_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("API_PORT", "8000")))
    parser.add_argument("--max-request-bytes", type=int, default=_default_request_limit())
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    uvicorn.run(
        create_app(args.package, max_request_bytes=args.max_request_bytes),
        host=args.host,
        port=args.port,
        access_log=True,
    )


if __name__ == "__main__":
    main()
