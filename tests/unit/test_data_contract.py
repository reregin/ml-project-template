"""Tests for deterministic Phase 4 schema enforcement and profiling."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_project_template.config import Settings
from ml_project_template.data_contract import (
    DataContractError,
    build_data_profile,
    contract_columns,
    prepare_contract_data,
)


def test_prepare_contract_data_normalizes_safe_values_and_order(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    raw = example_data.copy()
    raw["numeric_feature_1"] = raw["numeric_feature_1"].astype("string")
    raw["category_feature_2"] = "  " + raw["category_feature_2"] + " "

    prepared = prepare_contract_data(raw, settings)

    assert prepared.columns.tolist() == list(contract_columns(settings))
    assert prepared["numeric_feature_1"].dtype == "float64"
    assert prepared["category_feature_2"].str.startswith(" ").sum() == 0
    assert prepared[settings.target_column].dtype == "int8"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.assign(unexpected="value"), "undeclared columns"),
        (
            lambda frame: frame.assign(numeric_feature_1="not-a-number"),
            "non-numeric",
        ),
        (
            lambda frame: frame.assign(numeric_feature_2=np.inf),
            "infinite",
        ),
        (
            lambda frame: frame.assign(category_feature_1="  "),
            "blank values",
        ),
    ],
)
def test_prepare_contract_data_rejects_unsafe_values(
    example_data: pd.DataFrame,
    settings: Settings,
    mutation: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    invalid = mutation(example_data.copy())

    with pytest.raises(DataContractError, match=message):
        prepare_contract_data(invalid, settings)


def test_prepare_contract_data_rejects_duplicates_and_high_cardinality(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    duplicated = pd.concat([example_data, example_data.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataContractError, match="duplicate rows"):
        prepare_contract_data(duplicated, settings)

    strict_settings = Settings(project_root=settings.project_root, max_category_cardinality=2)
    with pytest.raises(DataContractError, match="cardinality"):
        prepare_contract_data(example_data, strict_settings)


def test_time_contract_normalizes_utc_and_profile_is_json_safe(
    example_data: pd.DataFrame,
    tmp_path: Path,
) -> None:
    settings = Settings(
        project_root=tmp_path,
        split_strategy="time",
        time_column="event_time",
    )
    raw = example_data.copy()
    raw["event_time"] = pd.date_range("2025-01-01", periods=len(raw), freq="h")

    prepared = prepare_contract_data(raw, settings)
    profile = build_data_profile(prepared, settings)

    assert prepared.columns[-2:].tolist() == ["event_time", "target"]
    assert prepared["event_time"].str.endswith("Z").all()
    assert profile["contract_version"] == "1"
    assert profile["rows"] == len(raw)
    assert profile["categorical_cardinality"] == {
        "category_feature_1": 3,
        "category_feature_2": 2,
    }
