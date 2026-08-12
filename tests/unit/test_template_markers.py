"""Tests for declared template customization-marker discovery."""

from pathlib import Path

from scripts.check_template_markers import declared_marker_occurrences


def test_every_manifest_marker_is_searchable_and_reported(tmp_path: Path) -> None:
    (tmp_path / "template-manifest.json").write_text(
        '{"identity_tokens":{"old_name":"identity"},'
        '"customization_markers":{"TODO_OWNER":"owner"}}',
        encoding="utf-8",
    )
    (tmp_path / "README.md").write_text("old_name TODO_OWNER", encoding="utf-8")

    assert declared_marker_occurrences(tmp_path) == {
        "old_name": ["README.md"],
        "TODO_OWNER": ["README.md"],
    }
