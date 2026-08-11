"""Train and persist the authoritative end-to-end model pipeline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml_project_template.config import Settings, configure_logging, get_settings
from ml_project_template.data import load_training_data
from ml_project_template.evaluation import split_development_test
from ml_project_template.lineage import read_data_manifest
from ml_project_template.pipeline import build_pipeline

LOGGER = logging.getLogger(__name__)


def train_model(
    data_path: Path,
    artifact_path: Path,
    metrics_path: Path,
    settings: Settings,
    data_manifest_path: Path | None = None,
) -> tuple[Any, dict[str, float | int | str]]:
    """Fit on the training split, evaluate once, and save the complete pipeline."""

    data = load_training_data(data_path, settings, data_manifest_path)
    features = data.loc[:, list(settings.feature_columns)]
    target = data[settings.target_column]
    timestamps = data[settings.time_column] if settings.time_column else None
    split = split_development_test(
        features,
        target,
        settings,
        timestamps,
    )

    model = build_pipeline(settings)
    model.fit(split.X_development, split.y_development)

    predictions = model.predict(split.X_test)
    probabilities = model.predict_proba(split.X_test)[:, 1]
    metrics: dict[str, float | int | str] = {
        "accuracy": float(accuracy_score(split.y_test, predictions)),
        "precision": float(precision_score(split.y_test, predictions, zero_division=0)),
        "recall": float(recall_score(split.y_test, predictions, zero_division=0)),
        "f1": float(f1_score(split.y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(split.y_test, probabilities)),
        "training_rows": int(len(split.X_development)),
        "test_rows": int(len(split.X_test)),
        "split_strategy": settings.split_strategy,
    }
    if data_manifest_path is not None:
        manifest = read_data_manifest(data_manifest_path)
        metrics["data_reference"] = str(manifest["output"]["immutable_reference"])

    resolved_artifact_path = artifact_path.expanduser().resolve()
    resolved_metrics_path = metrics_path.expanduser().resolve()
    resolved_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_metrics_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, resolved_artifact_path)
    resolved_metrics_path.write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    LOGGER.info(
        "event=model_trained artifact=%s training_rows=%d test_rows=%d f1=%.4f",
        resolved_artifact_path,
        len(split.X_development),
        len(split.X_test),
        metrics["f1"],
    )
    return model, metrics


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.prepared_data_path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--artifact", type=Path, default=settings.model_artifact_path)
    parser.add_argument("--metrics", type=Path, default=settings.metrics_path)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    settings = get_settings()
    args = parse_args()
    data_manifest = args.data_manifest
    if data_manifest is None and args.data.expanduser().resolve() == settings.prepared_data_path:
        data_manifest = settings.data_manifest_path
    _, metrics = train_model(
        args.data,
        args.artifact,
        args.metrics,
        settings,
        data_manifest,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
