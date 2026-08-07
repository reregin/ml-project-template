"""Generate a concise model card for the selected Phase 3 model."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def render_model_card(
    *,
    selected_candidate: str,
    run_id: str,
    data_sha256: str,
    metrics: dict[str, float | int],
    quality_gate_min_f1: float,
    best_params: dict[str, Any],
) -> str:
    """Render documented evidence without claiming real-world validity."""

    parameters = "\n".join(f"- `{name}`: `{value}`" for name, value in sorted(best_params.items()))
    if not parameters:
        parameters = "- Default baseline parameters"

    return f"""# Generated Model Card

## Model details

- Selected candidate: `{selected_candidate}`
- MLflow run ID: `{run_id}`
- Training data SHA-256: `{data_sha256}`
- Task: tabular binary classification
- Selection rule: highest development cross-validation F1

## Final untouched-test metrics

| Metric | Value |
|---|---:|
| Accuracy | {float(metrics["test_accuracy"]):.4f} |
| Precision | {float(metrics["test_precision"]):.4f} |
| Recall | {float(metrics["test_recall"]):.4f} |
| F1 | {float(metrics["test_f1"]):.4f} |
| ROC AUC | {float(metrics["test_roc_auc"]):.4f} |

Quality gate: test F1 must be at least `{quality_gate_min_f1:.4f}`.

## Selected parameters

{parameters}

## Intended use

This artifact proves the template's experiment, selection, tracking, and
inference contracts using domain-neutral synthetic data. It is not intended for
real decisions.

## Limitations

- The dataset is synthetic and deliberately small.
- The initial task supports binary classification only.
- The metrics do not establish fairness, calibration, robustness, or business
  value.
- Production data quality, drift, privacy, security, and delayed ground truth are
  not covered by this phase.
"""


def write_model_card(path: Path, content: str) -> Path:
    """Write the generated card and return its resolved path."""

    resolved_path = path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(content, encoding="utf-8")
    return resolved_path
