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
from sklearn.model_selection import train_test_split

from ml_project_template.config import Settings, configure_logging, get_settings
from ml_project_template.data import load_training_data
from ml_project_template.pipeline import build_pipeline


LOGGER = logging.getLogger(__name__)


def train_model(
    data_path: Path,
    artifact_path: Path,
    metrics_path: Path,
    settings: Settings,
) -> tuple[Any, dict[str, float | int]]:
    """Fit on the training split, evaluate once, and save the complete pipeline."""

    data = load_training_data(data_path, settings)
    features = data.loc[:, list(settings.feature_columns)]
    target = data[settings.target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=target,
    )

    model = build_pipeline(settings)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics: dict[str, float | int] = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "f1": float(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
    }

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
        len(X_train),
        len(X_test),
        metrics["f1"],
    )
    return model, metrics


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.raw_data_path)
    parser.add_argument("--artifact", type=Path, default=settings.model_artifact_path)
    parser.add_argument("--metrics", type=Path, default=settings.metrics_path)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    settings = get_settings()
    args = parse_args()
    _, metrics = train_model(args.data, args.artifact, args.metrics, settings)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
