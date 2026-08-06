"""Construction of the raw-input-to-prediction baseline pipeline."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_project_template.config import Settings


def build_pipeline(settings: Settings) -> Pipeline:
    """Build an unfitted, leakage-safe binary-classification pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(settings.numeric_features)),
            (
                "categorical",
                categorical_pipeline,
                list(settings.categorical_features),
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "estimator",
                LogisticRegression(
                    max_iter=1000,
                    random_state=settings.random_state,
                ),
            ),
        ]
    )
