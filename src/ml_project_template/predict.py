"""Load the authoritative model artifact and predict from one raw JSON record."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ml_project_template.config import configure_logging, get_settings

LOGGER = logging.getLogger(__name__)


def load_model(artifact_path: Path) -> Any:
    """Load the exact artifact produced by the training command."""

    resolved_path = artifact_path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Model artifact not found at {resolved_path}. Run training first.")
    model = joblib.load(resolved_path)
    LOGGER.info("event=model_loaded artifact=%s", resolved_path)
    return model


def validate_record(record: dict[str, Any], feature_columns: list[str]) -> None:
    missing_columns = [column for column in feature_columns if column not in record]
    if missing_columns:
        raise ValueError(f"Prediction input is missing columns: {missing_columns}")


def predict_record(record: dict[str, Any], artifact_path: Path) -> dict[str, Any]:
    """Predict one raw record using embedded fitted preprocessing."""

    model = load_model(artifact_path)
    feature_columns = [str(column) for column in model.feature_names_in_]
    validate_record(record, feature_columns)
    frame = pd.DataFrame([record], columns=feature_columns)

    prediction = model.predict(frame)[0]
    probability: float | None = None
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 in classes:
            positive_class_index = classes.index(1)
            probability = float(model.predict_proba(frame)[0][positive_class_index])

    result = {
        "prediction": int(prediction),
        "probability": probability if probability is not None else None,
    }
    LOGGER.info("event=prediction_complete prediction=%s", result["prediction"])
    return result


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, default=settings.model_artifact_path)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-json",
        help="A JSON object containing one raw feature record.",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Path to a JSON file containing one raw feature record.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    if args.input_file is not None:
        record = json.loads(args.input_file.read_text(encoding="utf-8"))
    else:
        record = json.loads(args.input_json)
    if not isinstance(record, dict):
        raise ValueError("Prediction input must decode to a JSON object.")
    print(json.dumps(predict_record(record, args.artifact), indent=2))


if __name__ == "__main__":
    main()
