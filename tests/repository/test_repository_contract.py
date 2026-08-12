"""Fast checks for repository-level template promises."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_notebooks_are_valid_json() -> None:
    notebooks = sorted((PROJECT_ROOT / "notebooks").glob("*.ipynb"))
    assert notebooks, "The learning track must include at least one notebook."

    for notebook_path in notebooks:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4
        assert isinstance(notebook["cells"], list)


def test_authoritative_code_has_no_runtime_path_mutation() -> None:
    source_files = (PROJECT_ROOT / "src" / "ml_project_template").glob("*.py")
    combined_source = "\n".join(path.read_text(encoding="utf-8") for path in source_files)

    assert "sys.path" not in combined_source
    assert "from src" not in combined_source
    assert "import src" not in combined_source


def test_sample_request_uses_exact_configured_features() -> None:
    from ml_project_template.config import get_settings

    request_path = PROJECT_ROOT / "examples" / "binary_classification" / "sample_request.json"
    request = json.loads(request_path.read_text(encoding="utf-8"))

    assert list(request) == list(get_settings().feature_columns)


def test_data_workflow_uses_shared_command_and_artifact_upload() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "data-pipeline.yml").read_text(
        encoding="utf-8"
    )

    assert "python -m ml_project_template.prepare_data" in workflow
    assert "verify_data_manifest" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "git commit" not in workflow


def test_package_workflow_uses_shared_sha_addressed_build_without_publishing() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "train-package.yml").read_text(
        encoding="utf-8"
    )

    assert "python -m ml_project_template.model_package" in workflow
    assert "ml-template-validate-package" in workflow
    assert "model-package-${{ github.sha }}" in workflow
    assert "actions/upload-artifact@v4" in workflow
    assert "docker build" in workflow
    assert "smoke_service.py --mode invalid" in workflow
    assert "compare_service_prediction.py" in workflow
    assert "docker push" not in workflow
    assert "packages: write" not in workflow


def test_container_contract_is_non_root_read_only_and_model_external() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    compose = (PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8")
    dockerignore = (PROJECT_ROOT / ".dockerignore").read_text(encoding="utf-8")

    assert dockerfile.startswith("FROM python:3.12.11-slim-bookworm")
    assert "USER 10001:10001" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert "COPY artifacts" not in dockerfile
    assert "read_only: true" in compose
    assert "no-new-privileges:true" in compose
    assert "cap_drop:" in compose
    assert "artifacts" in dockerignore
    assert "password" not in compose.lower()
