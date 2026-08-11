"""Tests for shared splitting, metrics, and diagnostics."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_project_template.config import Settings
from ml_project_template.evaluation import (
    calculate_classification_metrics,
    diagnostic_payloads,
    split_development_test,
)
from ml_project_template.pipeline import build_pipeline


def test_split_reserves_stratified_test_set(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    features = example_data.loc[:, list(settings.feature_columns)]
    target = example_data[settings.target_column]

    split = split_development_test(features, target, settings)

    assert len(split.X_development) == 96
    assert len(split.X_test) == 24
    assert split.y_development.value_counts().to_dict() == {0: 48, 1: 48}
    assert split.y_test.value_counts().to_dict() == {0: 12, 1: 12}
    assert set(split.X_development.index).isdisjoint(split.X_test.index)


def test_metrics_and_diagnostics_are_json_serializable(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    features = example_data.loc[:, list(settings.feature_columns)]
    target = example_data[settings.target_column]
    model = build_pipeline(settings).fit(features, target)

    metrics, predictions, probabilities = calculate_classification_metrics(
        model,
        features,
        target,
    )
    report, matrix = diagnostic_payloads(target, predictions)

    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert all(0 <= value <= 1 for value in metrics.values())
    assert len(predictions) == len(example_data)
    assert len(probabilities) == len(example_data)
    assert set(report) >= {"0", "1", "accuracy", "macro avg", "weighted avg"}
    assert len(matrix) == 2
    assert all(len(row) == 2 for row in matrix)


def test_time_split_puts_newest_rows_in_test_without_shuffling(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        split_strategy="time",
        time_column="event_time",
        cv_folds=2,
    )
    timestamps = pd.Series(pd.date_range("2025-01-01", periods=40, freq="D"))
    features = pd.DataFrame({"value": range(40)})
    target = pd.Series([0, 1] * 20)
    shuffled = pd.Series(range(40)).sample(frac=1, random_state=7).tolist()

    split = split_development_test(
        features.iloc[shuffled],
        target.iloc[shuffled],
        settings,
        timestamps.iloc[shuffled],
    )

    assert len(split.X_test) == 8
    assert split.X_development["value"].max() == 31
    assert split.X_test["value"].min() == 32


def test_time_split_requires_valid_timestamps_and_both_test_classes(tmp_path: Path) -> None:
    settings = Settings(
        project_root=tmp_path,
        split_strategy="time",
        time_column="event_time",
        cv_folds=2,
    )
    features = pd.DataFrame({"value": range(20)})
    target = pd.Series([0, 1] * 8 + [1] * 4)

    with pytest.raises(ValueError, match="timestamps are required"):
        split_development_test(features, target, settings)

    with pytest.raises(ValueError, match="test split"):
        split_development_test(
            features,
            target,
            settings,
            pd.Series(pd.date_range("2025-01-01", periods=20)),
        )
