"""Tests for generated model-card evidence."""

from pathlib import Path

from ml_project_template.model_card import render_model_card, write_model_card


def test_render_and_write_model_card(tmp_path: Path) -> None:
    content = render_model_card(
        selected_candidate="tuned",
        run_id="run-123",
        data_sha256="abc123",
        metrics={
            "test_accuracy": 0.8,
            "test_precision": 0.81,
            "test_recall": 0.79,
            "test_f1": 0.8,
            "test_roc_auc": 0.9,
        },
        quality_gate_min_f1=0.7,
        best_params={"estimator__C": 1.0},
    )

    output_path = write_model_card(tmp_path / "nested" / "model-card.md", content)

    assert output_path.is_file()
    assert "tuned" in content
    assert "run-123" in content
    assert "0.8000" in content
    assert "estimator__C" in content


def test_model_card_describes_default_parameters_when_empty() -> None:
    content = render_model_card(
        selected_candidate="baseline",
        run_id="run-123",
        data_sha256="abc123",
        metrics={
            "test_accuracy": 0.8,
            "test_precision": 0.8,
            "test_recall": 0.8,
            "test_f1": 0.8,
            "test_roc_auc": 0.8,
        },
        quality_gate_min_f1=0.7,
        best_params={},
    )

    assert "Default baseline parameters" in content
