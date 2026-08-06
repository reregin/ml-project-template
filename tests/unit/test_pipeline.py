"""Tests for raw feature preprocessing and estimator composition."""

import pandas as pd

from ml_project_template.config import Settings
from ml_project_template.pipeline import build_pipeline


def test_pipeline_fits_raw_data_and_handles_unknown_category(
    example_data: pd.DataFrame,
    settings: Settings,
) -> None:
    features = example_data.loc[:, list(settings.feature_columns)]
    target = example_data[settings.target_column]
    model = build_pipeline(settings)
    model.fit(features, target)

    record = features.iloc[[0]].copy()
    record.loc[:, "category_feature_1"] = "never_seen_during_fit"

    prediction = model.predict(record)

    assert list(model.named_steps) == ["preprocessor", "estimator"]
    assert model.feature_names_in_.tolist() == list(settings.feature_columns)
    assert prediction.shape == (1,)
    assert int(prediction[0]) in {0, 1}
