"""Focused tests for bounded monitoring outcomes and output labels."""

from ml_project_template.monitoring import ServiceMetrics


def test_operational_status_mapping_and_bounded_output_fallback() -> None:
    metrics = ServiceMetrics()

    for status in (200, 400, 413, 422, 503, 500):
        metrics.observe_response(status)
    metrics.observe_prediction(7)

    assert metrics.responses.labels(outcome="success")._value.get() == 1
    assert metrics.responses.labels(outcome="invalid")._value.get() == 3
    assert metrics.responses.labels(outcome="unavailable")._value.get() == 1
    assert metrics.responses.labels(outcome="error")._value.get() == 1
    assert metrics.outputs.labels(**{"class": "other"})._value.get() == 1
