"""Tests for the dependency-free metrics contract checker."""

from scripts.check_metrics import REQUIRED_METRICS, missing_metrics


def test_missing_metrics_reports_only_absent_contract_names() -> None:
    payload = "\n".join(f"# TYPE {name} gauge" for name in sorted(REQUIRED_METRICS))

    assert missing_metrics(payload) == set()
    assert missing_metrics("ml_model_ready 1") == REQUIRED_METRICS - {"ml_model_ready"}
