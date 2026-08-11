"""Validate raw CSV data and produce deterministic prepared-data evidence."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pandas as pd

from ml_project_template.config import Settings, configure_logging, get_settings
from ml_project_template.data_contract import (
    build_data_profile,
    contract_snapshot,
    prepare_contract_data,
)
from ml_project_template.lineage import git_revision, sha256_file, verify_data_manifest

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparationResult:
    """Paths and immutable identities produced by data preparation."""

    data_path: Path
    manifest_path: Path
    profile_path: Path
    input_reference: str
    output_reference: str

    def as_dict(self) -> dict[str, str]:
        return {
            "data_path": str(self.data_path),
            "manifest_path": str(self.manifest_path),
            "profile_path": str(self.profile_path),
            "input_reference": self.input_reference,
            "output_reference": self.output_reference,
        }


def _write_text_atomically(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temporary_directory:
        temporary_path = Path(temporary_directory) / path.name
        temporary_path.write_text(content, encoding="utf-8", newline="")
        temporary_path.replace(path)
    return path


def _write_json_atomically(path: Path, payload: dict[str, Any]) -> Path:
    return _write_text_atomically(
        path,
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def _load_raw_csv(path: Path) -> pd.DataFrame:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Raw data not found at {resolved_path}.")
    with resolved_path.open("r", encoding="utf-8-sig", newline="") as source:
        header = next(csv.reader(source), [])
    duplicate_headers = sorted({column for column in header if header.count(column) > 1})
    if duplicate_headers:
        raise ValueError(f"Raw CSV contains duplicate columns: {duplicate_headers}")
    return pd.read_csv(resolved_path)


def prepare_data(
    *,
    input_path: Path,
    output_path: Path,
    manifest_path: Path,
    profile_path: Path,
    settings: Settings,
    source_uri: str | None = None,
) -> PreparationResult:
    """Create canonical data, a profile, and content-addressed provenance."""

    resolved_input = input_path.expanduser().resolve()
    resolved_output = output_path.expanduser().resolve()
    resolved_manifest = manifest_path.expanduser().resolve()
    resolved_profile = profile_path.expanduser().resolve()
    raw_data = _load_raw_csv(resolved_input)
    prepared_data = prepare_contract_data(raw_data, settings)

    _write_text_atomically(
        resolved_output,
        prepared_data.to_csv(index=False, lineterminator="\n"),
    )
    validated_output = prepare_contract_data(pd.read_csv(resolved_output), settings)
    if validated_output.shape != prepared_data.shape:
        raise RuntimeError("Prepared output validation changed the dataset shape.")

    profile = build_data_profile(prepared_data, settings)
    _write_json_atomically(resolved_profile, profile)
    input_hash = sha256_file(resolved_input)
    output_hash = sha256_file(resolved_output)
    input_reference = f"sha256:{input_hash}"
    output_reference = f"sha256:{output_hash}"
    manifest = {
        "manifest_version": 1,
        "contract": contract_snapshot(settings),
        "preparation": {
            "implementation": "ml_project_template.prepare_data",
            "code_revision": git_revision(settings.project_root),
        },
        "input": {
            "path": str(resolved_input),
            "sha256": input_hash,
            "immutable_reference": input_reference,
            "source_uri": source_uri,
        },
        "output": {
            "path": str(resolved_output),
            "sha256": output_hash,
            "immutable_reference": output_reference,
        },
        "profile": {
            "path": str(resolved_profile),
            "sha256": sha256_file(resolved_profile),
        },
    }
    _write_json_atomically(resolved_manifest, manifest)
    verify_data_manifest(resolved_output, resolved_manifest)

    result = PreparationResult(
        data_path=resolved_output,
        manifest_path=resolved_manifest,
        profile_path=resolved_profile,
        input_reference=input_reference,
        output_reference=output_reference,
    )
    LOGGER.info(
        "event=data_prepared input=%s output=%s rows=%d reference=%s",
        resolved_input,
        resolved_output,
        len(prepared_data),
        output_reference,
    )
    return result


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=settings.raw_data_path)
    parser.add_argument("--output", type=Path, default=settings.prepared_data_path)
    parser.add_argument("--manifest", type=Path, default=settings.data_manifest_path)
    parser.add_argument("--profile", type=Path, default=settings.data_profile_path)
    parser.add_argument(
        "--source-uri",
        help="Optional immutable upstream URI, such as an object version or dataset tag.",
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    args = parse_args()
    result = prepare_data(
        input_path=args.input,
        output_path=args.output,
        manifest_path=args.manifest,
        profile_path=args.profile,
        settings=get_settings(),
        source_uri=args.source_uri,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
