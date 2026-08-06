"""Shared deterministic fixtures for unit and integration tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ml_project_template.config import Settings
from ml_project_template.example_data import generate_example_data


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(project_root=tmp_path)


@pytest.fixture
def example_data() -> pd.DataFrame:
    return generate_example_data(rows=120, random_state=42)
