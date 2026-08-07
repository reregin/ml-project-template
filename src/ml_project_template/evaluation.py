"""Shared classification evaluation and development/test splitting."""

from __future__ import annotations

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
) -> DatasetSplit:
    """Reserve the test set before any model comparison or tuning."""

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
