"""Tests for training-data loading and minimum contract validation."""

from pathlib import Path

import pandas as pd
import pytest

from ml_project_template.config import Settings
from ml_project_template.data import load_training_data, validate_training_data


def test_validate_training_data_accepts_example(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    validate_training_data(example_data, settings)


def test_validate_training_data_rejects_empty(settings: Settings) -> None:
    with pytest.raises(ValueError, match="at least one row"):
        validate_training_data(pd.DataFrame(), settings)


def test_validate_training_data_rejects_missing_column(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    invalid_data = example_data.drop(columns=["numeric_feature_2"])

    with pytest.raises(ValueError, match="numeric_feature_2"):
        validate_training_data(invalid_data, settings)


def test_validate_training_data_rejects_missing_target_value(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    invalid_data = example_data.copy()
    invalid_data.loc[0, settings.target_column] = None

    with pytest.raises(ValueError, match="target column"):
        validate_training_data(invalid_data, settings)


@pytest.mark.parametrize("target_value", [0, 1, 2])
def test_validate_training_data_rejects_invalid_target_classes(
    example_data: pd.DataFrame,
    settings: Settings,
    target_value: int,
) -> None:
    invalid_data = example_data.copy()
    invalid_data[settings.target_column] = target_value

    with pytest.raises(ValueError, match="binary target"):
        validate_training_data(invalid_data, settings)


def test_load_training_data_rejects_missing_file(
    tmp_path: Path,
    settings: Settings,
) -> None:
    with pytest.raises(FileNotFoundError, match="Generate the example data first"):
        load_training_data(tmp_path / "missing.csv", settings)


def test_load_training_data_reads_valid_csv(
    tmp_path: Path,
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    data_path = tmp_path / "input.csv"
    example_data.to_csv(data_path, index=False)

    loaded = load_training_data(data_path, settings)

    assert loaded.shape == example_data.shape
    assert loaded.columns.tolist() == example_data.columns.tolist()
