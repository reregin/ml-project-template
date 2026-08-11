"""Build and validate an immutable, SHA-addressable model package."""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast

import joblib
import mlflow.pyfunc
import mlflow.sklearn
import pandas as pd
from mlflow.models import Model, infer_signature

from ml_project_template.config import Settings, configure_logging, get_settings
from ml_project_template.experiment import ExperimentResult, run_experiment
from ml_project_template.lineage import (
    git_revision,
    git_worktree_is_dirty,
    read_data_manifest,
    sha256_file,
)
from ml_project_template.predict import predict_record
from ml_project_template.tracking import serializable_settings

LOGGER = logging.getLogger(__name__)
PACKAGE_MANIFEST_VERSION = 1
SAFE_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
REQUIRED_PACKAGE_PATHS = {
    "data-manifest.json",
    "data-profile.json",
    "inference-request.json",
    "metrics.json",
    "model-card.md",
    "model.joblib",
    "model/MLmodel",
    "resolved-config.json",
}


class ModelPackageError(ValueError):
    """Raised when a build cannot produce a trustworthy model package."""


@dataclass(frozen=True)
class ModelPackageResult:
    """Identity and paths for a successfully validated package."""

    artifact_version: str
    package_path: Path
    package_manifest_path: Path
    selected_run_id: str
    model_uri: str

    def as_dict(self) -> dict[str, str]:
        return {
            "artifact_version": self.artifact_version,
            "package_path": str(self.package_path),
            "package_manifest_path": str(self.package_manifest_path),
            "selected_run_id": self.selected_run_id,
            "model_uri": self.model_uri,
        }


def resolve_artifact_version(settings: Settings, requested_version: str | None) -> str:
    """Resolve an explicit version, CI commit SHA, Git revision, or local label."""

    ci_version = os.getenv("GITHUB_SHA")
    revision = git_revision(settings.project_root)
    version = requested_version or ci_version or revision
    if version == "unavailable":
        version = "local"
    if (
        requested_version is None
        and ci_version is None
        and git_worktree_is_dirty(settings.project_root)
    ):
        raise ModelPackageError(
            "The Git worktree has uncommitted changes; commit them or provide an explicit "
            "local --version."
        )
    if not SAFE_VERSION.fullmatch(version):
        raise ModelPackageError(
            "Artifact version must use only letters, digits, periods, underscores, and hyphens."
        )
    return version


def _read_request(path: Path, settings: Settings) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise ModelPackageError(f"Inference request not found at {resolved_path}.")
    try:
        request = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ModelPackageError(f"Inference request is not valid JSON: {resolved_path}.") from error
    if not isinstance(request, dict):
        raise ModelPackageError("Inference request must be a JSON object.")
    if list(request) != list(settings.feature_columns):
        raise ModelPackageError(
            "Inference request fields and order must exactly match configured features."
        )
    return request


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="",
    )


def _package_files(package_path: Path) -> dict[str, dict[str, Any]]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(package_path.rglob("*")):
        relative_path = path.relative_to(package_path).as_posix()
        if path.is_file() and relative_path != "package-manifest.json":
            files[relative_path] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    return files


def _save_mlflow_model(
    *,
    model_path: Path,
    joblib_path: Path,
    request: dict[str, Any],
    experiment_result: ExperimentResult,
    artifact_version: str,
    data_reference: str,
) -> None:
    model = joblib.load(joblib_path)
    input_example = pd.DataFrame([request])
    prediction = model.predict(input_example)
    signature = infer_signature(input_example, prediction)
    mlflow.sklearn.save_model(
        sk_model=model,
        path=str(model_path),
        signature=signature,
        input_example=input_example,
        serialization_format="cloudpickle",
        metadata={
            "artifact_version": artifact_version,
            "selected_run_id": experiment_result.selected_run_id,
            "tracked_model_uri": experiment_result.model_uri,
            "data_reference": data_reference,
        },
    )


def validate_model_package(package_path: Path) -> dict[str, Any]:
    """Fail if required outputs, hashes, model contract, or smoke inference break."""

    resolved_package = package_path.expanduser().resolve()
    manifest_path = resolved_package / "package-manifest.json"
    if not manifest_path.is_file():
        raise ModelPackageError(f"Package manifest not found at {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        recorded_files = manifest["files"]
        artifact_version = manifest["artifact_version"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ModelPackageError("Package manifest is invalid.") from error
    if manifest.get("manifest_version") != PACKAGE_MANIFEST_VERSION:
        raise ModelPackageError("Unsupported package manifest version.")
    if artifact_version != resolved_package.name:
        raise ModelPackageError("Package directory must match its artifact version.")
    if (
        re.fullmatch(r"[0-9a-fA-F]{40}", artifact_version)
        and manifest.get("code_revision") != artifact_version
    ):
        raise ModelPackageError("Commit-SHA package version does not match its code revision.")

    actual_files = _package_files(resolved_package)
    if set(actual_files) != set(recorded_files):
        raise ModelPackageError("Package file inventory does not match its manifest.")
    missing_required = sorted(REQUIRED_PACKAGE_PATHS - set(actual_files))
    if missing_required:
        raise ModelPackageError(f"Package is missing required outputs: {missing_required}")
    for relative_path, actual in actual_files.items():
        if actual != recorded_files[relative_path]:
            raise ModelPackageError(f"Package file failed integrity validation: {relative_path}")

    metrics = json.loads((resolved_package / "metrics.json").read_text(encoding="utf-8"))
    if metrics.get("quality_gate_passed") is not True:
        raise ModelPackageError("Package metrics do not contain a passed quality gate.")
    if metrics.get("data_reference") != manifest.get("data_reference"):
        raise ModelPackageError("Package metrics and manifest disagree on data identity.")

    mlflow_metadata = Model.load(str(resolved_package / "model" / "MLmodel"))
    if mlflow_metadata.signature is None:
        raise ModelPackageError("Packaged MLflow model has no signature.")
    if mlflow_metadata.saved_input_example_info is None:
        raise ModelPackageError("Packaged MLflow model has no input example.")

    request = json.loads((resolved_package / "inference-request.json").read_text(encoding="utf-8"))
    local_prediction = predict_record(request, resolved_package / "model.joblib")
    mlflow_model = mlflow.pyfunc.load_model(str(resolved_package / "model"))
    mlflow_prediction = mlflow_model.predict(pd.DataFrame([request]))
    if len(mlflow_prediction) != 1 or local_prediction["prediction"] not in {0, 1}:
        raise ModelPackageError("Packaged model failed its inference smoke test.")
    return cast(dict[str, Any], manifest)


def build_model_package(
    *,
    data_path: Path,
    data_manifest_path: Path,
    data_profile_path: Path,
    request_path: Path,
    output_root: Path,
    settings: Settings,
    tracking_uri: str,
    experiment_name: str,
    quality_gate_min_f1: float,
    artifact_version: str | None = None,
) -> ModelPackageResult:
    """Run the authoritative experiment and atomically publish a valid bundle."""

    resolved_version = resolve_artifact_version(settings, artifact_version)
    resolved_output_root = output_root.expanduser().resolve()
    final_package_path = resolved_output_root / resolved_version
    if final_package_path.exists():
        raise FileExistsError(
            f"Immutable package already exists at {final_package_path}; use a new version."
        )
    request = _read_request(request_path, settings)
    resolved_profile_path = data_profile_path.expanduser().resolve()
    if not resolved_profile_path.is_file():
        raise ModelPackageError(f"Data profile not found at {resolved_profile_path}.")
    data_manifest = read_data_manifest(data_manifest_path)
    data_reference = str(data_manifest["output"]["immutable_reference"])
    resolved_output_root.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=resolved_output_root) as temporary_directory:
        staging_path = Path(temporary_directory) / resolved_version
        staging_path.mkdir()
        result = run_experiment(
            data_path=data_path,
            data_manifest_path=data_manifest_path,
            artifact_path=staging_path / "model.joblib",
            metrics_path=staging_path / "metrics.json",
            model_card_path=staging_path / "model-card.md",
            settings=settings,
            tracking_uri=tracking_uri,
            experiment_name=experiment_name,
            quality_gate_min_f1=quality_gate_min_f1,
        )
        shutil.copy2(request_path.expanduser().resolve(), staging_path / "inference-request.json")
        shutil.copy2(data_manifest_path.expanduser().resolve(), staging_path / "data-manifest.json")
        shutil.copy2(resolved_profile_path, staging_path / "data-profile.json")
        _write_json(staging_path / "resolved-config.json", serializable_settings(settings))
        _save_mlflow_model(
            model_path=staging_path / "model",
            joblib_path=staging_path / "model.joblib",
            request=request,
            experiment_result=result,
            artifact_version=resolved_version,
            data_reference=data_reference,
        )
        package_manifest = {
            "manifest_version": PACKAGE_MANIFEST_VERSION,
            "artifact_version": resolved_version,
            "code_revision": git_revision(settings.project_root),
            "code_worktree_dirty": git_worktree_is_dirty(settings.project_root),
            "data_reference": data_reference,
            "selected_candidate": result.selected_candidate,
            "selected_run_id": result.selected_run_id,
            "tracked_model_uri": result.model_uri,
            "files": _package_files(staging_path),
        }
        _write_json(staging_path / "package-manifest.json", package_manifest)
        validate_model_package(staging_path)
        staging_path.replace(final_package_path)

    package_result = ModelPackageResult(
        artifact_version=resolved_version,
        package_path=final_package_path,
        package_manifest_path=final_package_path / "package-manifest.json",
        selected_run_id=result.selected_run_id,
        model_uri=result.model_uri,
    )
    LOGGER.info(
        "event=model_package_complete version=%s path=%s run_id=%s",
        resolved_version,
        final_package_path,
        result.selected_run_id,
    )
    return package_result


def parse_build_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.prepared_data_path)
    parser.add_argument("--data-manifest", type=Path, default=settings.data_manifest_path)
    parser.add_argument("--data-profile", type=Path, default=settings.data_profile_path)
    parser.add_argument(
        "--request",
        type=Path,
        default=settings.project_root
        / "examples"
        / "binary_classification"
        / "sample_request.json",
    )
    parser.add_argument("--output-root", type=Path, default=settings.model_package_root)
    parser.add_argument("--version")
    parser.add_argument("--tracking-uri", default=settings.tracking_uri)
    parser.add_argument("--experiment-name", default=f"{settings.experiment_name}-package")
    parser.add_argument("--min-test-f1", type=float, default=settings.quality_gate_min_f1)
    return parser.parse_args()


def build_main() -> None:
    configure_logging()
    settings = get_settings()
    args = parse_build_args()
    result = build_model_package(
        data_path=args.data,
        data_manifest_path=args.data_manifest,
        data_profile_path=args.data_profile,
        request_path=args.request,
        output_root=args.output_root,
        settings=settings,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        quality_gate_min_f1=args.min_test_f1,
        artifact_version=args.version,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


def parse_validate_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a Phase 5 model package.")
    parser.add_argument("--package", type=Path, required=True)
    return parser.parse_args()


def validate_main() -> None:
    configure_logging()
    args = parse_validate_args()
    manifest = validate_model_package(args.package)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    build_main()
