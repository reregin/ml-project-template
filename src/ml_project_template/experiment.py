"""Compare baseline and tuned candidates, track evidence, and package the winner."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate

from ml_project_template.config import Settings, configure_logging, get_settings
from ml_project_template.data import load_training_data
from ml_project_template.evaluation import (
    calculate_classification_metrics,
    diagnostic_payloads,
    split_development_test,
)
from ml_project_template.model_card import render_model_card, write_model_card
from ml_project_template.pipeline import build_pipeline
from ml_project_template.tracking import (
    configure_tracking,
    git_revision,
    log_reproducibility_artifacts,
    sha256_file,
)

LOGGER = logging.getLogger(__name__)
SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


class QualityGateError(RuntimeError):
    """Raised when the selected model must not become a deployable artifact."""


@dataclass(frozen=True)
class CandidateResult:
    name: str
    run_id: str
    selection_score: float
    cv_metrics: dict[str, float]
    best_params: dict[str, Any]
    estimator: Any


@dataclass(frozen=True)
class ExperimentResult:
    selected_candidate: str
    selected_run_id: str
    candidate_run_ids: dict[str, str]
    selection_scores: dict[str, float]
    test_metrics: dict[str, float]
    model_uri: str
    artifact_path: Path
    metrics_path: Path
    model_card_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate": self.selected_candidate,
            "selected_run_id": self.selected_run_id,
            "candidate_run_ids": self.candidate_run_ids,
            "selection_scores": self.selection_scores,
            "test_metrics": self.test_metrics,
            "model_uri": self.model_uri,
            "artifact_path": str(self.artifact_path),
            "metrics_path": str(self.metrics_path),
            "model_card_path": str(self.model_card_path),
        }


def _cross_validation(settings: Settings) -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=settings.cv_folds,
        shuffle=True,
        random_state=settings.random_state,
    )


def _mean_std_metrics(cv_results: dict[str, Any]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for metric_name in SCORING:
        scores = cv_results[f"test_{metric_name}"]
        metrics[f"cv_mean_{metric_name}"] = float(scores.mean())
        metrics[f"cv_std_{metric_name}"] = float(scores.std())
    return metrics


def _candidate_parameters(
    settings: Settings,
    candidate_name: str,
    data_sha256: str,
    code_revision: str,
) -> dict[str, Any]:
    return {
        "candidate": candidate_name,
        "selection_metric": settings.selection_metric,
        "cv_folds": settings.cv_folds,
        "test_size": settings.test_size,
        "random_state": settings.random_state,
        "data_sha256": data_sha256,
        "code_revision": code_revision,
    }


def run_baseline_candidate(
    *,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    settings: Settings,
    data: pd.DataFrame,
    data_sha256: str,
    code_revision: str,
) -> CandidateResult:
    """Track the default pipeline using development-only cross-validation."""

    estimator = build_pipeline(settings)
    cv_results = cross_validate(
        estimator,
        X_development,
        y_development,
        cv=_cross_validation(settings),
        scoring=SCORING,
        return_train_score=False,
    )
    metrics = _mean_std_metrics(cv_results)
    best_params = {
        "estimator__C": 1.0,
        "estimator__class_weight": None,
    }

    with mlflow.start_run(run_name="baseline") as run:
        mlflow.log_params(
            _candidate_parameters(
                settings,
                "baseline",
                data_sha256,
                code_revision,
            )
        )
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "candidate": "baseline",
                "selection_status": "candidate",
                "evaluation_scope": "development_cross_validation",
            }
        )
        log_reproducibility_artifacts(
            settings=settings,
            data=data,
            data_sha256=data_sha256,
            code_revision=code_revision,
        )
        return CandidateResult(
            name="baseline",
            run_id=run.info.run_id,
            selection_score=metrics["cv_mean_f1"],
            cv_metrics=metrics,
            best_params=best_params,
            estimator=estimator,
        )


def run_tuned_candidate(
    *,
    X_development: pd.DataFrame,
    y_development: pd.Series,
    settings: Settings,
    data: pd.DataFrame,
    data_sha256: str,
    code_revision: str,
) -> CandidateResult:
    """Tune only within development data and track the complete search evidence."""

    parameter_grid = {
        "estimator__C": [0.1, 1.0, 10.0],
        "estimator__class_weight": [None, "balanced"],
    }
    search = GridSearchCV(
        estimator=build_pipeline(settings),
        param_grid=parameter_grid,
        scoring=SCORING,
        refit=settings.selection_metric,
        cv=_cross_validation(settings),
        return_train_score=True,
        n_jobs=1,
    )
    search.fit(X_development, y_development)
    best_index = int(search.best_index_)
    metrics: dict[str, float] = {}
    for metric_name in SCORING:
        metrics[f"cv_mean_{metric_name}"] = float(
            search.cv_results_[f"mean_test_{metric_name}"][best_index]
        )
        metrics[f"cv_std_{metric_name}"] = float(
            search.cv_results_[f"std_test_{metric_name}"][best_index]
        )
    cv_results_frame = pd.DataFrame(search.cv_results_).sort_values(
        f"rank_test_{settings.selection_metric}"
    )

    with mlflow.start_run(run_name="tuned") as run:
        mlflow.log_params(
            _candidate_parameters(
                settings,
                "tuned",
                data_sha256,
                code_revision,
            )
        )
        mlflow.log_params(
            {
                "search_type": "GridSearchCV",
                "search_candidates": len(cv_results_frame),
                **search.best_params_,
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                "candidate": "tuned",
                "selection_status": "candidate",
                "evaluation_scope": "development_cross_validation",
            }
        )
        log_reproducibility_artifacts(
            settings=settings,
            data=data,
            data_sha256=data_sha256,
            code_revision=code_revision,
            extra_artifacts={"cv_results.csv": cv_results_frame},
        )
        return CandidateResult(
            name="tuned",
            run_id=run.info.run_id,
            selection_score=metrics["cv_mean_f1"],
            cv_metrics=metrics,
            best_params=dict(search.best_params_),
            estimator=search.best_estimator_,
        )


def _select_candidate(candidates: list[CandidateResult]) -> CandidateResult:
    """Select by development CV only; the test set is unavailable here."""

    return max(candidates, key=lambda candidate: candidate.selection_score)


def _write_metrics(path: Path, payload: dict[str, Any]) -> Path:
    resolved_path = path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return resolved_path


def run_experiment(
    *,
    data_path: Path,
    artifact_path: Path,
    metrics_path: Path,
    model_card_path: Path,
    settings: Settings,
    tracking_uri: str,
    experiment_name: str,
    quality_gate_min_f1: float,
) -> ExperimentResult:
    """Run comparison, select without test leakage, gate, log, and package."""

    if not 0 <= quality_gate_min_f1 <= 1:
        raise ValueError("quality_gate_min_f1 must be between 0 and 1.")

    configure_tracking(tracking_uri, experiment_name)
    resolved_data_path = data_path.expanduser().resolve()
    data = load_training_data(resolved_data_path, settings)
    features = data.loc[:, list(settings.feature_columns)]
    target = data[settings.target_column]
    split = split_development_test(features, target, settings)
    data_digest = sha256_file(resolved_data_path)
    revision = git_revision(settings.project_root)

    candidates = [
        run_baseline_candidate(
            X_development=split.X_development,
            y_development=split.y_development,
            settings=settings,
            data=data,
            data_sha256=data_digest,
            code_revision=revision,
        ),
        run_tuned_candidate(
            X_development=split.X_development,
            y_development=split.y_development,
            settings=settings,
            data=data,
            data_sha256=data_digest,
            code_revision=revision,
        ),
    ]
    selected = _select_candidate(candidates)
    selected_model = clone(selected.estimator)
    selected_model.fit(split.X_development, split.y_development)
    final_metrics, predictions, _ = calculate_classification_metrics(
        selected_model,
        split.X_test,
        split.y_test,
    )
    test_metrics = {f"test_{name}": value for name, value in final_metrics.items()}
    classification_payload, confusion_payload = diagnostic_payloads(
        split.y_test,
        predictions,
    )
    gate_passed = test_metrics["test_f1"] >= quality_gate_min_f1
    client = MlflowClient()

    for candidate in candidates:
        client.set_tag(
            candidate.run_id,
            "selection_status",
            "selected" if candidate.run_id == selected.run_id else "not_selected",
        )

    selection_payload = {
        "selection_metric": settings.selection_metric,
        "selected_candidate": selected.name,
        "candidate_scores": {candidate.name: candidate.selection_score for candidate in candidates},
        "test_used_during_selection": False,
        "quality_gate_min_f1": quality_gate_min_f1,
        "quality_gate_passed": gate_passed,
    }

    with mlflow.start_run(run_id=selected.run_id):
        mlflow.log_metrics(test_metrics)
        mlflow.log_metric("quality_gate_passed", float(gate_passed))
        mlflow.set_tags(
            {
                "quality_gate": "passed" if gate_passed else "failed",
                "final_test_evaluated": "true",
            }
        )
        log_reproducibility_artifacts(
            settings=settings,
            data=data,
            data_sha256=data_digest,
            code_revision=revision,
            extra_artifacts={
                "selection.json": selection_payload,
                "classification_report.json": classification_payload,
                "confusion_matrix.json": confusion_payload,
            },
        )

        if not gate_passed:
            raise QualityGateError(
                "Selected model failed the packaging quality gate: "
                f"test_f1={test_metrics['test_f1']:.4f} < {quality_gate_min_f1:.4f}."
            )

        input_example = split.X_development.head(5)
        signature = infer_signature(
            input_example,
            selected_model.predict(input_example),
        )
        model_info = mlflow.sklearn.log_model(
            sk_model=selected_model,
            name="model",
            signature=signature,
            input_example=input_example,
            serialization_format="cloudpickle",
            metadata={
                "selected_candidate": selected.name,
                "data_sha256": data_digest,
            },
        )

        card_content = render_model_card(
            selected_candidate=selected.name,
            run_id=selected.run_id,
            data_sha256=data_digest,
            metrics=test_metrics,
            quality_gate_min_f1=quality_gate_min_f1,
            best_params=selected.best_params,
        )
        resolved_card_path = write_model_card(model_card_path, card_content)
        mlflow.log_artifact(str(resolved_card_path), artifact_path="evidence")

    resolved_artifact_path = artifact_path.expanduser().resolve()
    resolved_artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, resolved_artifact_path)

    metrics_payload: dict[str, Any] = {
        **selection_payload,
        **test_metrics,
        "selected_run_id": selected.run_id,
        "candidate_run_ids": {candidate.name: candidate.run_id for candidate in candidates},
        "data_sha256": data_digest,
        "code_revision": revision,
        "model_uri": model_info.model_uri,
    }
    resolved_metrics_path = _write_metrics(metrics_path, metrics_payload)
    result = ExperimentResult(
        selected_candidate=selected.name,
        selected_run_id=selected.run_id,
        candidate_run_ids={candidate.name: candidate.run_id for candidate in candidates},
        selection_scores={candidate.name: candidate.selection_score for candidate in candidates},
        test_metrics=test_metrics,
        model_uri=model_info.model_uri,
        artifact_path=resolved_artifact_path,
        metrics_path=resolved_metrics_path,
        model_card_path=resolved_card_path,
    )
    LOGGER.info(
        "event=experiment_complete selected=%s run_id=%s test_f1=%.4f model_uri=%s",
        result.selected_candidate,
        result.selected_run_id,
        result.test_metrics["test_f1"],
        result.model_uri,
    )
    return result


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=settings.raw_data_path)
    parser.add_argument("--artifact", type=Path, default=settings.model_artifact_path)
    parser.add_argument("--metrics", type=Path, default=settings.metrics_path)
    parser.add_argument("--model-card", type=Path, default=settings.model_card_path)
    parser.add_argument("--tracking-uri", default=settings.tracking_uri)
    parser.add_argument("--experiment-name", default=settings.experiment_name)
    parser.add_argument(
        "--min-test-f1",
        type=float,
        default=settings.quality_gate_min_f1,
    )
    return parser.parse_args()


def main() -> None:
    configure_logging()
    settings = get_settings()
    args = parse_args()
    result = run_experiment(
        data_path=args.data,
        artifact_path=args.artifact,
        metrics_path=args.metrics,
        model_card_path=args.model_card,
        settings=settings,
        tracking_uri=args.tracking_uri,
        experiment_name=args.experiment_name,
        quality_gate_min_f1=args.min_test_f1,
    )
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
