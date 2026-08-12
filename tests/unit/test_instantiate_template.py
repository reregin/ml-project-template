"""Prove a fresh renamed project operates without the template checkout."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.instantiate_template import instantiate_template, validate_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_names_and_destination_are_guarded(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="project-name"):
        validate_names("Unsafe Name", "safe_name")
    with pytest.raises(ValueError, match="package-name"):
        validate_names("safe-name", "unsafe-name")
    with pytest.raises(ValueError, match="different"):
        validate_names("ml-project-template", "ml_project_template")
    with pytest.raises(ValueError, match="outside"):
        instantiate_template(
            PROJECT_ROOT,
            PROJECT_ROOT / "nested",
            project_name="safe-name",
            package_name="safe_name",
        )
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        instantiate_template(
            PROJECT_ROOT,
            existing,
            project_name="safe-name",
            package_name="safe_name",
        )


def test_fresh_project_runs_core_path_without_template_imports(tmp_path: Path) -> None:
    destination = instantiate_template(
        PROJECT_ROOT,
        tmp_path / "fresh-neutral-project",
        project_name="fresh-neutral-project",
        package_name="fresh_neutral_project",
    )
    assert (destination / "src" / "fresh_neutral_project" / "train.py").is_file()
    assert not (destination / "src" / "ml_project_template").exists()
    assert not (destination / ".git").exists()
    assert not (destination / ".venv").exists()
    assert not (destination / "scripts" / "instantiate_template.py").exists()
    assert not (destination / "tests" / "unit" / "test_instantiate_template.py").exists()
    assert "TEMPLATE_MAINTAINER_ONLY" not in (
        destination / ".github" / "workflows" / "ci.yml"
    ).read_text(encoding="utf-8")

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(destination / "src")
    environment["ML_PROJECT_ROOT"] = str(destination)
    commands = [
        [sys.executable, "-m", "fresh_neutral_project.example_data", "--rows", "40"],
        [sys.executable, "-m", "fresh_neutral_project.prepare_data"],
        [sys.executable, "-m", "fresh_neutral_project.train"],
        [
            sys.executable,
            "-m",
            "fresh_neutral_project.predict",
            "--input-file",
            "examples/binary_classification/sample_request.json",
        ],
    ]
    outputs = [
        subprocess.run(
            command,
            cwd=destination,
            env=environment,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for command in commands
    ]
    prediction = json.loads(outputs[-1])
    assert prediction["prediction"] in {0, 1}
    assert "ml_project_template" not in (destination / "pyproject.toml").read_text(encoding="utf-8")
