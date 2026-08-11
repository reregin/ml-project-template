"""Shared classification evaluation and development/test splitting."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from ml_project_template.config import Settings


@dataclass(frozen=True)
class DatasetSplit:
    """Development data used for selection and untouched final test data."""

    X_development: pd.DataFrame
    X_test: pd.DataFrame
    y_development: pd.Series
    y_test: pd.Series


def split_development_test(
    features: pd.DataFrame,
    target: pd.Series,
    settings: Settings,
    timestamps: pd.Series | None = None,
) -> DatasetSplit:
    """Reserve the test set using the configured random or temporal boundary."""

    if settings.split_strategy == "time":
        if timestamps is None:
            raise ValueError("timestamps are required for a time-based split.")
        parsed_timestamps = pd.to_datetime(timestamps, errors="coerce", utc=True)
        if parsed_timestamps.isna().any():
            raise ValueError("timestamps must be parseable and non-missing.")
        ordered_positions = parsed_timestamps.sort_values(kind="mergesort").index
        ordered_features = features.loc[ordered_positions]
        ordered_target = target.loc[ordered_positions]
        test_rows = math.ceil(len(ordered_features) * settings.test_size)
        boundary = len(ordered_features) - test_rows
        X_development = ordered_features.iloc[:boundary]
        X_test = ordered_features.iloc[boundary:]
        y_development = ordered_target.iloc[:boundary]
        y_test = ordered_target.iloc[boundary:]
        for name, values in (("development", y_development), ("test", y_test)):
            if set(values.unique()) != {0, 1}:
                raise ValueError(f"The time-based {name} split must contain both target classes.")
        if int(y_development.value_counts().min()) < settings.cv_folds:
            raise ValueError(
                "The time-based development split has too few rows per class "
                "for the configured cross-validation."
            )
        return DatasetSplit(X_development, X_test, y_development, y_test)

    X_development, X_test, y_development, y_test = train_test_split(
        features,
        target,
        test_size=settings.test_size,
        random_state=settings.random_state,
        stratify=target,
    )
    return DatasetSplit(X_development, X_test, y_development, y_test)


def calculate_classification_metrics(
    model: Any,
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[dict[str, float], list[int], list[float]]:
    """Calculate the authoritative binary-classification metric set."""

    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
    }
    return metrics, predictions.astype(int).tolist(), probabilities.astype(float).tolist()


def diagnostic_payloads(
    target: pd.Series,
    predictions: list[int],
) -> tuple[dict[str, Any], list[list[int]]]:
    """Return JSON-serializable classification diagnostics."""

    report = classification_report(
        target,
        predictions,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(target, predictions).astype(int).tolist()
    return report, matrix
