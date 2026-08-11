"""Tests for idempotent prepared-data artifacts and manifest integrity."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml_project_template.config import Settings
from ml_project_template.example_data import write_example_data
from ml_project_template.lineage import DataLineageError, verify_data_manifest
from ml_project_template.prepare_data import prepare_data


def _paths(tmp_path: Path) -> dict[str, Path]:
    return {
        "input_path": tmp_path / "raw" / "input.csv",
        "output_path": tmp_path / "processed" / "training.csv",
        "manifest_path": tmp_path / "processed" / "manifest.json",
        "profile_path": tmp_path / "processed" / "profile.json",
    }


def test_prepare_data_is_idempotent_and_content_addressed(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    paths = _paths(tmp_path)
    write_example_data(paths["input_path"], rows=80, random_state=settings.random_state)

    first = prepare_data(
        **paths,
        settings=settings,
        source_uri="s3://example-bucket/dataset.csv?versionId=immutable",
    )
    first_bytes = {name: path.read_bytes() for name, path in paths.items() if name != "input_path"}
    second = prepare_data(
        **paths, settings=settings, source_uri="s3://example-bucket/dataset.csv?versionId=immutable"
    )

    assert first == second
    assert all(
        path.read_bytes() == first_bytes[name]
        for name, path in paths.items()
        if name != "input_path"
    )
    manifest = verify_data_manifest(first.data_path, first.manifest_path)
    assert manifest["input"]["immutable_reference"] == first.input_reference
    assert manifest["output"]["immutable_reference"] == first.output_reference
    assert manifest["input"]["source_uri"].endswith("versionId=immutable")
    assert json.loads(first.profile_path.read_text(encoding="utf-8"))["rows"] == 80


def test_manifest_verification_rejects_tampered_output(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path)
    paths = _paths(tmp_path)
    write_example_data(paths["input_path"], rows=40, random_state=settings.random_state)
    result = prepare_data(**paths, settings=settings)
    tampered = pd.read_csv(result.data_path)
    tampered.loc[0, "numeric_feature_1"] = 999
    tampered.to_csv(result.data_path, index=False)

    with pytest.raises(DataLineageError, match="does not match"):
        verify_data_manifest(result.data_path, result.manifest_path)


def test_prepare_data_rejects_duplicate_csv_headers(tmp_path: Path) -> None:
    settings = Settings(project_root=tmp_path, minimum_rows=2, cv_folds=2, test_size=0.5)
    paths = _paths(tmp_path)
    paths["input_path"].parent.mkdir(parents=True)
    paths["input_path"].write_text(
        "numeric_feature_1,numeric_feature_1,category_feature_1,category_feature_2,target\n"
        "1,2,a,b,0\n2,3,a,b,1\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate columns"):
        prepare_data(**paths, settings=settings)
