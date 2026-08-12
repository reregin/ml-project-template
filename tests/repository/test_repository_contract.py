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
    assert "promtool" in workflow
    assert "alerts.test.yml" in workflow
    assert "check_metrics.py" in workflow
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
    assert "${GRAFANA_ADMIN_PASSWORD:?" in compose
    assert "admin/admin" not in compose.lower()


def test_monitoring_stack_is_pinned_provisioned_and_persistent() -> None:
    import yaml

    compose = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text(encoding="utf-8"))
    prometheus = yaml.safe_load(
        (PROJECT_ROOT / "monitoring" / "prometheus" / "prometheus.yml").read_text(encoding="utf-8")
    )
    datasource = yaml.safe_load(
        (
            PROJECT_ROOT
            / "monitoring"
            / "grafana"
            / "provisioning"
            / "datasources"
            / "prometheus.yml"
        ).read_text(encoding="utf-8")
    )
    dashboard = json.loads(
        (PROJECT_ROOT / "monitoring" / "grafana" / "dashboards" / "ml-service.json").read_text(
            encoding="utf-8"
        )
    )

    services = compose["services"]
    assert services["prometheus"]["image"] == "prom/prometheus:v3.13.1"
    assert services["grafana"]["image"] == "grafana/grafana:13.1.0"
    assert "prometheus-data:/prometheus" in services["prometheus"]["volumes"]
    assert prometheus["scrape_configs"][0]["static_configs"][0]["targets"] == [
        "prediction-api:8000"
    ]
    assert datasource["datasources"][0]["uid"] == "prometheus"
    assert dashboard["uid"] == "ml-prediction-service"
    assert len(dashboard["panels"]) >= 8


def test_alerts_have_rule_tests_and_runbook_actions() -> None:
    import yaml

    alerts_path = PROJECT_ROOT / "monitoring" / "prometheus" / "alerts.yml"
    tests_path = PROJECT_ROOT / "monitoring" / "prometheus" / "alerts.test.yml"
    runbook_path = PROJECT_ROOT / "docs" / "operations-runbook.md"
    alerts = yaml.safe_load(alerts_path.read_text(encoding="utf-8"))
    rule_tests = yaml.safe_load(tests_path.read_text(encoding="utf-8"))
    runbook = runbook_path.read_text(encoding="utf-8")
    names = {rule["alert"] for rule in alerts["groups"][0]["rules"]}
    tested_names = {
        expectation["alertname"]
        for test in rule_tests["tests"]
        for expectation in test["alert_rule_test"]
    }

    assert names == {
        "MLModelUnavailable",
        "MLPredictionErrorRateHigh",
        "MLPredictionLatencyHigh",
    }
    assert tested_names == names
    assert "## Model unavailable" in runbook
    assert "## High prediction error rate" in runbook
    assert "## High prediction latency" in runbook


def test_metric_labels_are_bounded_and_privacy_safe() -> None:
    monitoring_source = (PROJECT_ROOT / "src" / "ml_project_template" / "monitoring.py").read_text(
        encoding="utf-8"
    )

    assert 'labelnames=("outcome",)' in monitoring_source
    assert 'labelnames=("class",)' in monitoring_source
    for forbidden_label in ("user_id", "request_id", "run_id", "artifact_version", "feature"):
        assert f'"{forbidden_label}"' not in monitoring_source


def test_progressive_tracks_and_customization_manifest_are_complete() -> None:
    customization = (PROJECT_ROOT / "docs" / "template-customization.md").read_text(
        encoding="utf-8"
    )
    manifest = json.loads((PROJECT_ROOT / "template-manifest.json").read_text(encoding="utf-8"))

    assert "### DS-only track" in customization
    assert "### ML-engineering track" in customization
    assert "### Full MLOps reference track" in customization
    for required in (
        "project name",
        "package name",
        "target",
        "task type",
        "schema",
        "metrics",
        "threshold",
        "ports",
        "registry",
        "owners",
    ):
        assert required in customization.lower()
    assert set(manifest["identity_tokens"]) == {
        "ML Project Template",
        "ml-project-template",
        "ml_project_template",
    }


def test_release_security_and_sbom_automation_is_declared() -> None:
    release = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    package = (PROJECT_ROOT / ".github" / "workflows" / "train-package.yml").read_text(
        encoding="utf-8"
    )
    dependency_review = (
        PROJECT_ROOT / ".github" / "workflows" / "dependency-review.yml"
    ).read_text(encoding="utf-8")

    assert 'tags:\n      - "v*.*.*"' in release
    assert "version mismatch" in release
    assert "python -m pip_audit" in release
    assert "format: cyclonedx" in release
    assert "gh release create" in release
    assert "trivy-action@ed142fd0673e97e23eac54620cfb913e5ce36c25" in package
    assert "severity: HIGH,CRITICAL" in package
    assert "serving-image-sbom-${{ github.sha }}" in package
    assert "actions/dependency-review-action@v4" in dependency_review
    assert "--ignore-vuln PYSEC-2026-3552" in release
    exceptions = (PROJECT_ROOT / "docs" / "security-exceptions.md").read_text(encoding="utf-8")
    assert "Review no later than:** 2026-09-12" in exceptions
    assert "cryptography 50" in exceptions


def test_template_versions_and_changelog_agree() -> None:
    import re

    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)
    assert version_match is not None
    project_version = version_match.group(1)
    package_init = (PROJECT_ROOT / "src" / "ml_project_template" / "__init__.py").read_text(
        encoding="utf-8"
    )
    template_version = json.loads(
        (PROJECT_ROOT / "template-manifest.json").read_text(encoding="utf-8")
    )["template_version"]
    changelog = (PROJECT_ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert project_version == template_version == "1.0.0"
    assert f'__version__ = "{project_version}"' in package_init
    assert f"## [{project_version}]" in changelog


def test_ci_verifies_a_standalone_generated_project() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "instantiate_template.py" in workflow
    assert "${RUNNER_TEMP}/fresh-neutral-project" in workflow
    assert 'cd "${RUNNER_TEMP}/fresh-neutral-project"' in workflow
    assert "python -m pytest" in workflow
