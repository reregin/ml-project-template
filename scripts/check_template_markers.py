"""Verify template identities and unresolved customization markers are declared."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCAN_SUFFIXES = {".json", ".md", ".py", ".toml", ".yaml", ".yml"}
IGNORED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "artifacts",
    "build",
    "dist",
    "htmlcov",
    "mlartifacts",
    "mlruns",
    "__pycache__",
}


def declared_marker_occurrences(project_root: Path) -> dict[str, list[str]]:
    manifest = json.loads((project_root / "template-manifest.json").read_text(encoding="utf-8"))
    markers = {**manifest["identity_tokens"], **manifest["customization_markers"]}
    occurrences = dict.fromkeys(markers, [])
    for path in project_root.rglob("*"):
        if (
            not path.is_file()
            or path.name == "template-manifest.json"
            or any(part in IGNORED_DIRECTORIES for part in path.parts)
        ):
            continue
        if path.suffix not in SCAN_SUFFIXES and path.name != ".env.example":
            continue
        content = path.read_text(encoding="utf-8")
        for marker in markers:
            if marker in content:
                occurrences[marker] = [*occurrences[marker], str(path.relative_to(project_root))]
    return occurrences


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    occurrences = declared_marker_occurrences(args.project_root.resolve())
    print(json.dumps(occurrences, indent=2, sort_keys=True))
    if args.strict and any(occurrences.values()):
        raise SystemExit("Declared template markers remain; complete customization first.")


if __name__ == "__main__":
    main()
