"""Wait for an HTTP readiness endpoint with a bounded deadline."""

from __future__ import annotations

import argparse
import time
import urllib.error
import urllib.request


def wait_until_ready(url: str, timeout: float, interval: float) -> None:
    deadline = time.monotonic() + timeout
    last_error = "no response"
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=min(interval, 2.0)) as response:  # nosec B310
                if response.status == 200:
                    return
                last_error = f"HTTP {response.status}"
        except (ConnectionError, urllib.error.URLError, TimeoutError) as error:
            last_error = type(error).__name__
        time.sleep(interval)
    raise TimeoutError(f"Service did not become ready within {timeout}s ({last_error}).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/ready")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args()
    wait_until_ready(args.url, args.timeout, args.interval)


if __name__ == "__main__":
    main()
