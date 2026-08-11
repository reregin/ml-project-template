"""Data loading and validation for the authoritative training path."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ml_project_template.config import Settings
from ml_project_template.data_contract import prepare_contract_data
from ml_project_template.lineage import verify_data_manifest

LOGGER = logging.getLogger(__name__)


def validate_training_data(data: pd.DataFrame, settings: Settings) -> None:
    """Validate the complete Phase 4 contract required before training."""

    prepare_contract_data(data, settings)


def load_training_data(
    path: Path,
    settings: Settings,
    manifest_path: Path | None = None,
) -> pd.DataFrame:
    """Load, optionally verify lineage, and normalize contract-valid CSV data."""

    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(
            f"Training data not found at {resolved_path}. Generate the example data first."
        )

    if manifest_path is not None:
        verify_data_manifest(resolved_path, manifest_path)
    data = prepare_contract_data(pd.read_csv(resolved_path), settings)
    LOGGER.info("event=data_loaded path=%s rows=%d columns=%d", resolved_path, *data.shape)
    return data
