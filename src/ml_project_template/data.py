"""Data loading and validation for the authoritative training path."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml_project_template.config import Settings


LOGGER = logging.getLogger(__name__)


def validate_training_data(data: pd.DataFrame, settings: Settings) -> None:
    """Validate the minimum contract required by the baseline pipeline."""

    if data.empty:
        raise ValueError("Training data must contain at least one row.")

    required_columns = {*settings.feature_columns, settings.target_column}
    missing_columns = sorted(required_columns.difference(data.columns))
    if missing_columns:
        raise ValueError(f"Training data is missing columns: {missing_columns}")

    if data[settings.target_column].isna().any():
        raise ValueError("The target column must not contain missing values.")

    target_values = set(data[settings.target_column].unique())
    if target_values != {0, 1}:
        raise ValueError(
            "The Phase 1 example expects a binary target containing both 0 and 1. "
            f"Received: {sorted(target_values, key=str)}"
        )


def load_training_data(path: Path, settings: Settings) -> pd.DataFrame:
    """Load a CSV dataset and apply the baseline data contract."""

    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {resolved_path}. Generate the example data first."
        )

    data = pd.read_csv(resolved_path)
    validate_training_data(data, settings)
    LOGGER.info("event=data_loaded path=%s rows=%d columns=%d", resolved_path, *data.shape)
    return data

