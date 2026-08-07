"""Typed settings and authoritative project paths."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_project_root() -> Path:
    configured_root = os.getenv("ML_PROJECT_ROOT")
    if configured_root:
        return Path(configured_root).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    """Configuration for the generic binary-classification baseline."""

    project_root: Path = field(default_factory=_default_project_root)
    target_column: str = "target"
    numeric_features: tuple[str, ...] = (
        "numeric_feature_1",
        "numeric_feature_2",
    )
    categorical_features: tuple[str, ...] = (
        "category_feature_1",
        "category_feature_2",
    )
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 3
    selection_metric: str = "f1"
    quality_gate_min_f1: float = 0.7
    experiment_name: str = "ml-project-template"

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_root", self.project_root.resolve())
        if not 0 < self.test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2.")
        if self.selection_metric != "f1":
            raise ValueError("Phase 3 currently supports selection_metric='f1' only.")
        if not 0 <= self.quality_gate_min_f1 <= 1:
            raise ValueError("quality_gate_min_f1 must be between 0 and 1.")
        if not self.target_column.strip():
            raise ValueError("target_column must not be empty.")
        if not self.feature_columns:
            raise ValueError("At least one feature must be configured.")
        if len(self.feature_columns) != len(set(self.feature_columns)):
            raise ValueError("Feature names must be unique.")

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return self.numeric_features + self.categorical_features

    @property
    def raw_data_path(self) -> Path:
        return self.project_root / "data" / "raw" / "example.csv"

    @property
    def model_artifact_path(self) -> Path:
        return self.project_root / "models" / "model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.project_root / "models" / "metrics.json"

    @property
    def model_card_path(self) -> Path:
        return self.project_root / "models" / "model-card.md"

    @property
    def tracking_uri(self) -> str:
        configured_uri = os.getenv("MLFLOW_TRACKING_URI")
        if configured_uri:
            return configured_uri
        database_path = (self.project_root / "mlflow.db").as_posix()
        return f"sqlite:///{database_path}"


def get_settings() -> Settings:
    """Return validated settings; kept as a function for future overrides."""

    return Settings()


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise key-value logs for command-line workflows."""

    logging.basicConfig(
        level=level,
        format="%(asctime)s level=%(levelname)s logger=%(name)s message=%(message)s",
    )
