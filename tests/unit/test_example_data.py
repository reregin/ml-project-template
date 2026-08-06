"""Tests for the deterministic, domain-neutral fixture generator."""

import pandas as pd
import pytest

from ml_project_template.example_data import generate_example_data


def test_example_data_is_deterministic() -> None:
    first = generate_example_data(rows=100, random_state=7)
    second = generate_example_data(rows=100, random_state=7)

    pd.testing.assert_frame_equal(first, second)


def test_example_data_has_expected_contract() -> None:
    data = generate_example_data(rows=100, random_state=42)

    assert data.columns.tolist() == [
        "numeric_feature_1",
        "numeric_feature_2",
        "category_feature_1",
        "category_feature_2",
        "target",
    ]
    assert data["target"].value_counts().to_dict() == {0: 50, 1: 50}
    assert data.drop(columns="target").isna().any().any()


def test_example_data_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError, match="at least 20"):
        generate_example_data(rows=19)
