"""Exercise normal, invalid, or burst prediction-service behavior."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


def send_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[int, dict[str, Any]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            return response.status, json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read().decode("utf-8"))


def run_scenario(
    *,
    url: str,
    payload: dict[str, Any],
    mode: str,
    count: int,
    timeout: float,
    interval: float,
) -> dict[str, Any]:
    """Run one requested traffic shape and return a machine-readable summary."""

    request_payload = payload if mode != "invalid" else {"unexpected": "field"}
    expected_status = 422 if mode == "invalid" else 200
    request_count = count if mode == "burst" else 1

    if mode == "burst":
        with ThreadPoolExecutor(max_workers=min(request_count, 20)) as executor:
            results = list(
                executor.map(
                    lambda _: send_json(url, request_payload, timeout),
                    range(request_count),
                )
            )
    else:
        results = []
        for _ in range(request_count):
            results.append(send_json(url, request_payload, timeout))
            if interval:
                time.sleep(interval)

    statuses = [status for status, _ in results]
    return {
        "mode": mode,
        "requests": request_count,
        "expected_status": expected_status,
        "successful_expectations": sum(status == expected_status for status in statuses),
        "statuses": statuses,
        "sample_response": results[0][1],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument(
        "--request",
        type=Path,
        default=Path("examples/binary_classification/sample_request.json"),
    )
    parser.add_argument("--mode", choices=("normal", "invalid", "burst"), default="normal")
    parser.add_argument("--count", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--interval", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count < 1:
        raise ValueError("count must be at least 1.")
    payload = json.loads(args.request.read_text(encoding="utf-8"))
    summary = run_scenario(
        url=args.url,
        payload=payload,
        mode=args.mode,
        count=args.count,
        timeout=args.timeout,
        interval=args.interval,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["successful_expectations"] != summary["requests"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
