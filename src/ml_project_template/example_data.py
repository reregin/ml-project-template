"""Generate deterministic, domain-neutral data for proving the template works."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ml_project_template.config import configure_logging, get_settings

LOGGER = logging.getLogger(__name__)


def generate_example_data(rows: int = 400, random_state: int = 42) -> pd.DataFrame:
    """Create a small mixed-type dataset with missing values and a binary target."""

    if rows < 20:
        raise ValueError("rows must be at least 20 so both split classes are usable.")

    rng = np.random.default_rng(random_state)
    numeric_feature_1 = rng.normal(loc=0.0, scale=1.0, size=rows)
    numeric_feature_2 = rng.uniform(low=-2.0, high=2.0, size=rows)
    category_feature_1 = rng.choice(
        ["category_a", "category_b", "category_c"],
        size=rows,
        p=[0.45, 0.35, 0.20],
    ).astype(object)
    category_feature_2 = rng.choice(
        ["group_1", "group_2"],
        size=rows,
    ).astype(object)

    score = (
        1.1 * numeric_feature_1
        - 0.7 * numeric_feature_2
        + np.where(category_feature_1 == "category_c", 0.8, 0.0)
        + np.where(category_feature_2 == "group_2", 0.35, 0.0)
        + rng.normal(loc=0.0, scale=0.7, size=rows)
    )
    target = (score > np.median(score)).astype("int8")

    missing_count = max(1, rows // 25)
    numeric_missing = rng.choice(rows, size=missing_count, replace=False)
    categorical_missing = rng.choice(rows, size=missing_count, replace=False)
    numeric_feature_1[numeric_missing] = np.nan
    category_feature_1[categorical_missing] = None

    return pd.DataFrame(
        {
            "numeric_feature_1": numeric_feature_1,
            "numeric_feature_2": numeric_feature_2,
            "category_feature_1": category_feature_1,
            "category_feature_2": category_feature_2,
            "target": target,
        }
    )


def write_example_data(output_path: Path, rows: int, random_state: int) -> Path:
    """Generate and write example data, returning the resolved output path."""

    resolved_path = output_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    generate_example_data(rows=rows, random_state=random_state).to_csv(
        resolved_path,
        index=False,
    )
    LOGGER.info("event=example_data_written path=%s rows=%d", resolved_path, rows)
    return resolved_path


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=settings.raw_data_path)
    parser.add_argument("--rows", type=int, default=400)
    parser.add_argument("--random-state", type=int, default=settings.random_state)
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    write_example_data(args.output, args.rows, args.random_state)


if __name__ == "__main__":
    main()
