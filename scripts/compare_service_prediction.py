"""Prove an HTTP service matches direct inference from the same MLflow Model."""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path
from typing import Any

import mlflow.pyfunc
import pandas as pd

from ml_project_template.model_package import validate_model_package


def service_prediction(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
        return json.loads(response.read().decode("utf-8"))


def compare_predictions(
    *,
    package_path: Path,
    request_path: Path,
    url: str,
    timeout: float,
) -> dict[str, Any]:
    """Compare direct and HTTP predictions plus immutable version identity."""

    manifest = validate_model_package(package_path)
    payload = json.loads(request_path.read_text(encoding="utf-8"))
    model = mlflow.pyfunc.load_model(str(package_path / "model"))
    direct_values = model.predict(pd.DataFrame([payload]))
    direct_prediction = int(
        direct_values.iloc[0] if hasattr(direct_values, "iloc") else direct_values[0]
    )
    response = service_prediction(url, payload, timeout)
    service_value = int(response["prediction"])
    service_version = str(response["model"]["artifact_version"])
    expected_version = str(manifest["artifact_version"])
    matches = direct_prediction == service_value and service_version == expected_version
    return {
        "matches": matches,
        "direct_prediction": direct_prediction,
        "service_prediction": service_value,
        "expected_version": expected_version,
        "service_version": service_version,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument(
        "--request",
        type=Path,
        default=Path("examples/binary_classification/sample_request.json"),
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--timeout", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = compare_predictions(
        package_path=args.package,
        request_path=args.request,
        url=args.url,
        timeout=args.timeout,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["matches"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
