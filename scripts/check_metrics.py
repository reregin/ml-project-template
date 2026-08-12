"""Check that a serving endpoint exposes the complete Phase 7 metric contract."""

from __future__ import annotations

import argparse
import urllib.request

REQUIRED_METRICS = {
    "ml_model_ready",
    "ml_prediction_duration_seconds_bucket",
    "ml_prediction_outputs_total",
    "ml_prediction_requests_in_progress",
    "ml_prediction_requests_total",
    "ml_prediction_responses_total",
    "ml_prediction_rows_total",
}


def fetch_metrics(url: str, timeout: float) -> str:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # nosec B310
        if response.status != 200:
            raise RuntimeError(f"Metrics endpoint returned HTTP {response.status}.")
        return response.read().decode("utf-8")


def missing_metrics(payload: str) -> set[str]:
    return {name for name in REQUIRED_METRICS if name not in payload}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/metrics")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    payload = fetch_metrics(args.url, args.timeout)
    missing = missing_metrics(payload)
    if missing:
        raise SystemExit(f"Missing required metrics: {sorted(missing)}")
    print(f"metric_contract_ok count={len(REQUIRED_METRICS)}")


if __name__ == "__main__":
    main()
