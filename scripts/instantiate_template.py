"""Create a renamed standalone project without copying local runtime state."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

EXCLUDED_DIRECTORIES = {
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
}
EXCLUDED_SUFFIXES = {".egg-info", "__pycache__"}
EXCLUDED_FILES = {".coverage", ".env", "mlflow.db"}
TEMPLATE_MAINTAINER_FILES = {
    Path("scripts/check_template_markers.py"),
    Path("scripts/instantiate_template.py"),
    Path("tests/repository/test_repository_contract.py"),
    Path("tests/unit/test_instantiate_template.py"),
    Path("tests/unit/test_template_markers.py"),
}
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def validate_names(project_name: str, package_name: str) -> None:
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", project_name):
        raise ValueError("project-name must be a lowercase hyphenated slug.")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", package_name):
        raise ValueError("package-name must be a valid lowercase Python package name.")
    if project_name == "ml-project-template" or package_name == "ml_project_template":
        raise ValueError("Choose identities different from the template defaults.")


def should_copy(relative_path: Path) -> bool:
    if relative_path in TEMPLATE_MAINTAINER_FILES:
        return False
    if any(
        part in EXCLUDED_DIRECTORIES or part.endswith(tuple(EXCLUDED_SUFFIXES))
        for part in relative_path.parts
    ):
        return False
    if relative_path.name in EXCLUDED_FILES or relative_path.suffix in {
        ".db-shm",
        ".db-wal",
        ".log",
        ".pyc",
    }:
        return False
    generated_output = len(relative_path.parts) >= 2 and (
        relative_path.parts[0] == "models"
        or (
            relative_path.parts[0] == "data"
            and relative_path.parts[1] in {"raw", "interim", "processed"}
        )
    )
    if generated_output:
        return relative_path.name == ".gitkeep"
    return True


def _without_maintainer_blocks(content: str) -> str:
    lines = content.splitlines(keepends=True)
    output: list[str] = []
    skipping = False
    for line in lines:
        if "TEMPLATE_MAINTAINER_ONLY_START" in line:
            skipping = True
            continue
        if "TEMPLATE_MAINTAINER_ONLY_END" in line:
            skipping = False
            continue
        if not skipping:
            output.append(line)
    return "".join(output)


def instantiate_template(
    source: Path,
    destination: Path,
    *,
    project_name: str,
    package_name: str,
) -> Path:
    """Copy and rename one project, refusing to merge into an existing path."""

    validate_names(project_name, package_name)
    resolved_source = source.expanduser().resolve()
    resolved_destination = destination.expanduser().resolve()
    if resolved_source == resolved_destination or resolved_source in resolved_destination.parents:
        raise ValueError("Destination must be outside the template repository.")
    if resolved_destination.exists():
        raise FileExistsError(f"Destination already exists: {resolved_destination}")

    display_name = " ".join(part.capitalize() for part in project_name.split("-"))
    replacements = {
        "ml-project-template": project_name,
        "ml_project_template": package_name,
        "ML Project Template": display_name,
    }

    for current_root, directories, filenames in os.walk(resolved_source):
        directories[:] = [
            directory
            for directory in directories
            if directory not in EXCLUDED_DIRECTORIES
            and not directory.endswith(tuple(EXCLUDED_SUFFIXES))
        ]
        current_path = Path(current_root)
        for filename in filenames:
            source_path = current_path / filename
            relative_path = source_path.relative_to(resolved_source)
            if not should_copy(relative_path):
                continue
            destination_parts = [
                package_name if part == "ml_project_template" else part
                for part in relative_path.parts
            ]
            destination_path = resolved_destination.joinpath(*destination_parts)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            if source_path.suffix.lower() in TEXT_SUFFIXES or source_path.name in {
                ".dockerignore",
                ".env.example",
                ".gitignore",
            }:
                content = source_path.read_text(encoding="utf-8")
                for old, new in replacements.items():
                    content = content.replace(old, new)
                content = _without_maintainer_blocks(content)
                destination_path.write_text(content, encoding="utf-8")
            else:
                shutil.copy2(source_path, destination_path)

    manifest_path = resolved_destination / "template-manifest.json"
    source_manifest = json.loads(
        (resolved_source / "template-manifest.json").read_text(encoding="utf-8")
    )
    manifest_path.write_text(
        json.dumps(
            {
                "project_name": project_name,
                "package_name": package_name,
                "template_version": source_manifest["template_version"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    generated_contract = resolved_destination / "tests" / "repository" / "test_project_identity.py"
    generated_contract.parent.mkdir(parents=True, exist_ok=True)
    generated_contract.write_text(
        '"""Identity and independence checks for the instantiated project."""\n\n'
        "from pathlib import Path\n\n"
        "PROJECT_ROOT = Path(__file__).resolve().parents[2]\n\n\n"
        "def test_project_identity_is_self_contained() -> None:\n"
        '    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")\n'
        f'    assert (PROJECT_ROOT / "src" / "{package_name}").is_dir()\n'
        f'    assert "{project_name}" in pyproject\n'
        '    assert not (PROJECT_ROOT / "src" / "ml_project_template").exists()\n',
        encoding="utf-8",
    )
    return resolved_destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--project-name", required=True)
    parser.add_argument("--package-name", required=True)
    parser.add_argument("--source", type=Path, default=Path.cwd())
    args = parser.parse_args()
    created = instantiate_template(
        args.source,
        args.destination,
        project_name=args.project_name,
        package_name=args.package_name,
    )
    print(f"created={created}")


if __name__ == "__main__":
    main()
