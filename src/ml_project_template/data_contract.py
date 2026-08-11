"""Explicit, deterministic data contract for the generic tabular example."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from ml_project_template.config import Settings


class DataContractError(ValueError):
    """Raised when input data cannot safely enter training."""


def contract_columns(settings: Settings) -> tuple[str, ...]:
    """Return the canonical prepared-data column order."""

    time_columns = (settings.time_column,) if settings.time_column else ()
    return (*settings.feature_columns, *time_columns, settings.target_column)


def contract_snapshot(settings: Settings) -> dict[str, Any]:
    """Return the versioned settings that determine prepared-data bytes."""

    return {
        "version": settings.data_contract_version,
        "numeric_features": list(settings.numeric_features),
        "categorical_features": list(settings.categorical_features),
        "target_column": settings.target_column,
        "time_column": settings.time_column,
        "minimum_rows": settings.minimum_rows,
        "max_category_cardinality": settings.max_category_cardinality,
        "allow_extra_columns": settings.allow_extra_columns,
    }


def _coerce_numeric(data: pd.DataFrame, settings: Settings) -> None:
    for column in settings.numeric_features:
        original = data[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise DataContractError(f"Numeric column '{column}' contains non-numeric values.")
        finite_values = converted.dropna().to_numpy(dtype=float)
        if not np.isfinite(finite_values).all():
            raise DataContractError(f"Numeric column '{column}' contains infinite values.")
        data[column] = converted.astype(float)


def _normalize_categories(data: pd.DataFrame, settings: Settings) -> None:
    for column in settings.categorical_features:
        normalized = data[column].astype("string").str.strip()
        if normalized.dropna().eq("").any():
            raise DataContractError(f"Categorical column '{column}' contains blank values.")
        cardinality = int(normalized.nunique(dropna=True))
        if cardinality > settings.max_category_cardinality:
            raise DataContractError(
                f"Categorical column '{column}' has cardinality {cardinality}; "
                f"maximum is {settings.max_category_cardinality}."
            )
        data[column] = normalized.astype(object).where(normalized.notna(), np.nan)


def _normalize_target(data: pd.DataFrame, settings: Settings) -> None:
    target = pd.to_numeric(data[settings.target_column], errors="coerce")
    if target.isna().any():
        raise DataContractError("The target column must contain numeric, non-missing values.")
    target_values = set(target.unique())
    if target_values != {0, 1}:
        raise DataContractError(
            "The binary target must contain both 0 and 1. "
            f"Received: {sorted(target_values, key=str)}"
        )
    minimum_class_rows = math.ceil(settings.cv_folds / (1 - settings.test_size))
    class_counts = target.value_counts()
    if int(class_counts.min()) < minimum_class_rows:
        raise DataContractError(
            "Each target class needs at least "
            f"{minimum_class_rows} rows for the configured split and cross-validation."
        )
    data[settings.target_column] = target.astype("int8")


def _normalize_time(data: pd.DataFrame, settings: Settings) -> None:
    if not settings.time_column:
        return
    timestamps = pd.to_datetime(data[settings.time_column], errors="coerce", utc=True)
    if timestamps.isna().any():
        raise DataContractError(
            f"Time column '{settings.time_column}' must contain parseable, non-missing timestamps."
        )
    data[settings.time_column] = timestamps.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def prepare_contract_data(raw_data: pd.DataFrame, settings: Settings) -> pd.DataFrame:
    """Validate and canonicalize non-learned fields without fitting transforms."""

    if raw_data.empty:
        raise DataContractError("Data must contain at least one row.")
    if len(raw_data) < settings.minimum_rows:
        raise DataContractError(
            f"Data has {len(raw_data)} rows; minimum is {settings.minimum_rows}."
        )
    if raw_data.columns.duplicated().any():
        duplicates = sorted(set(raw_data.columns[raw_data.columns.duplicated()].tolist()))
        raise DataContractError(f"Data contains duplicate columns: {duplicates}")

    expected = set(contract_columns(settings))
    actual = set(raw_data.columns)
    missing = sorted(expected - actual)
    if missing:
        raise DataContractError(f"Data is missing required columns: {missing}")
    extra = sorted(actual - expected)
    if extra and not settings.allow_extra_columns:
        raise DataContractError(f"Data contains undeclared columns: {extra}")

    data = raw_data.loc[:, list(contract_columns(settings))].copy()
    _coerce_numeric(data, settings)
    _normalize_categories(data, settings)
    _normalize_target(data, settings)
    _normalize_time(data, settings)
    if data.duplicated().any():
        raise DataContractError("Data contains duplicate rows.")
    return data


def _finite_stat(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def build_data_profile(data: pd.DataFrame, settings: Settings) -> dict[str, Any]:
    """Build a deterministic, JSON-safe profile for validation and lineage."""

    numeric_summary: dict[str, dict[str, float | None]] = {}
    for column in settings.numeric_features:
        values = data[column].dropna()
        numeric_summary[column] = {
            "min": _finite_stat(values.min()),
            "max": _finite_stat(values.max()),
            "mean": _finite_stat(values.mean()),
        }

    return {
        "contract_version": settings.data_contract_version,
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "column_order": data.columns.tolist(),
        "dtypes": {column: str(dtype) for column, dtype in data.dtypes.items()},
        "missing_by_column": {column: int(count) for column, count in data.isna().sum().items()},
        "categorical_cardinality": {
            column: int(data[column].nunique(dropna=True))
            for column in settings.categorical_features
        },
        "numeric_summary": numeric_summary,
        "target_distribution": {
            str(label): int(count)
            for label, count in data[settings.target_column].value_counts().sort_index().items()
        },
    }
