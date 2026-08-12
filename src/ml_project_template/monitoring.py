"""Privacy-safe, bounded-cardinality Prometheus metrics for model serving."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

OUTCOMES = ("success", "invalid", "unavailable", "error")
OUTPUT_CLASSES = ("0", "1", "other")


class ServiceMetrics:
    """Own an isolated registry so app factories and tests never collide."""

    def __init__(self) -> None:
        self.registry = CollectorRegistry(auto_describe=True)
        self.requests = Counter(
            "ml_prediction_requests_total",
            "Prediction requests received.",
            registry=self.registry,
        )
        self.responses = Counter(
            "ml_prediction_responses_total",
            "Prediction responses grouped by a bounded operational outcome.",
            labelnames=("outcome",),
            registry=self.registry,
        )
        self.duration = Histogram(
            "ml_prediction_duration_seconds",
            "Prediction request duration, including validation and inference.",
            buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
            registry=self.registry,
        )
        self.rows = Counter(
            "ml_prediction_rows_total",
            "Rows successfully predicted.",
            registry=self.registry,
        )
        self.outputs = Counter(
            "ml_prediction_outputs_total",
            "Successful predictions grouped by the bounded model output class.",
            labelnames=("class",),
            registry=self.registry,
        )
        self.in_progress = Gauge(
            "ml_prediction_requests_in_progress",
            "Prediction requests currently being processed.",
            registry=self.registry,
        )
        self.model_ready = Gauge(
            "ml_model_ready",
            "Whether the validated model package is loaded and ready (1 or 0).",
            registry=self.registry,
        )
        for outcome in OUTCOMES:
            self.responses.labels(outcome=outcome)
        for output_class in OUTPUT_CLASSES:
            self.outputs.labels(**{"class": output_class})
        self.model_ready.set(0)

    def observe_response(self, status_code: int) -> None:
        """Map HTTP status to one of four deliberately bounded outcomes."""

        if 200 <= status_code < 300:
            outcome = "success"
        elif status_code in {400, 413, 422}:
            outcome = "invalid"
        elif status_code == 503:
            outcome = "unavailable"
        else:
            outcome = "error"
        self.responses.labels(outcome=outcome).inc()

    def observe_prediction(self, prediction: int) -> None:
        """Record one successful row without recording any input feature."""

        self.rows.inc()
        output_class = str(prediction) if str(prediction) in OUTPUT_CLASSES else "other"
        self.outputs.labels(**{"class": output_class}).inc()
