# ML Project Template

An executable, progressively adoptable template for tabular machine-learning
projects. The current baseline covers a domain-neutral binary-classification
workflow from raw data through experiment comparison, a fitted model pipeline,
and raw-input inference.

The repository is intentionally layered:

- **DS track:** notebooks, exploration, preprocessing, training, and evaluation;
- **ML engineering track:** installable package, reproducible commands, and one
  authoritative training/inference artifact;
- **MLOps track:** CI-backed evolution toward packaging, serving, monitoring, and
  alerting.

Phase 1 established the reliable core, Phase 2 protects it with automated
quality checks, Phase 3 adds evidence-based model comparison with MLflow, and
Phase 4 adds deterministic data preparation with validated lineage.
Phase 5 turns the selected model into a validated, commit-addressable package
using the same entry point locally and in CI. Phase 6 serves that exact package
through a typed FastAPI boundary and a hardened, health-checked container.
This is not yet the complete MLOps template; see
[the extension plan](docs/project-extension-plan.md).

For detailed setup, operation, customization, and troubleshooting instructions,
use the [project handbook](docs/project-handbook.md).

## What Phase 1 proves

The included synthetic example verifies this contract:

```text
raw mixed-type data
        ↓
validated feature and target columns
        ↓
stratified train/test split
        ↓
preprocessing fitted on training data only
        ↓
one fitted sklearn Pipeline (preprocessing + estimator)
        ↓
models/model.joblib
        ↓
prediction from raw JSON using that exact artifact
```

The example uses only neutral names:

- `numeric_feature_1`
- `numeric_feature_2`
- `category_feature_1`
- `category_feature_2`
- `target`

It is a smoke-test fixture, not a recommended real-world feature set or model.

## Quick start

From the repository root, create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Generate the deterministic example data:

```powershell
python -m ml_project_template.example_data
```

Validate and prepare it. This writes canonical data, a profile, and a manifest
containing content-addressed input/output references:

```powershell
python -m ml_project_template.prepare_data
```

Train and save the simple Phase 1 preprocessing-and-estimator pipeline:

```powershell
python -m ml_project_template.train
```

To compare the baseline with a tuned candidate, select using development-only
cross-validation, evaluate the winner once on the untouched test set, and
package it only if the quality gate passes:

```powershell
python -m pip install -e ".[experiment]"
python -m ml_project_template.experiment
```

This creates two comparable MLflow runs plus the selected local model, metrics,
and model card. Inspect the runs in another terminal:

```powershell
mlflow ui --backend-store-uri sqlite:///D:/projects/ml-project-template/mlflow.db
```

Then open `http://127.0.0.1:5000`. If the repository is elsewhere, replace the
absolute database path. See the handbook for remote tracking and quality-gate
configuration.

Build the complete Phase 5 model package. A clean committed revision becomes
the default version; `--version` is useful for an explicit local label:

```powershell
python -m ml_project_template.model_package --version local-check
ml-template-validate-package --package artifacts/model-packages/local-check
```

The bundle includes a standard MLflow Model, compatibility joblib model,
metrics, model card, inference request, data/configuration evidence, and a
SHA-256 inventory. Existing version directories are never overwritten.

Serve a validated package locally:

```powershell
python -m pip install -e ".[serve]"
ml-template-serve --package artifacts/model-packages/local-check
```

Then call `GET /live`, `GET /ready`, or `POST /predict` on
`http://127.0.0.1:8000`. To exercise the container instead, copy `.env.example`
to `.env`, set `MODEL_PACKAGE_HOST_PATH` to the absolute package directory, and
run `docker compose up --build`. The model package is mounted read-only rather
than copied into the reusable image.

Predict from one raw record. The committed request deliberately includes a
category that was not present during training:

```powershell
python -m ml_project_template.predict --input-file examples/binary_classification/sample_request.json
```

For short programmatic calls, `--input-json` also accepts an inline JSON object.

Successful training creates ignored local outputs:

```text
data/raw/example.csv
data/processed/training.csv
data/processed/manifest.json
data/processed/profile.json
models/model.joblib
models/metrics.json
```

## Optional notebook and application dependencies

Install experimentation tools:

```powershell
python -m pip install -e ".[experiment]"
```

Install and run the optional Streamlit example:

```powershell
python -m pip install -e ".[app]"
streamlit run app/main.py
```

For compatibility, `python -m pip install -r requirements.txt` installs both
optional groups. Core workflows should prefer the smaller `pip install -e .`.

## Quality checks

Install the development group:

```powershell
python -m pip install -e ".[dev]"
```

Run the complete local quality gate on Windows:

```powershell
PowerShell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

The gate checks formatting, lint, types, common security issues, 83 behavioral
tests, and a 90% minimum branch-coverage threshold. GitHub Actions runs the same
checks on Python 3.10 and 3.12 for pushes and pull requests.

## Current structure

```text
├── app/                              # optional Streamlit client
├── data/
│   ├── raw/                          # generated/local raw data
│   ├── processed/                    # validated data, manifest, and profile
│   └── external/                     # third-party reference data
├── docs/
│   ├── current-state-assessment.md
│   └── project-extension-plan.md
├── examples/
│   └── binary_classification/
│       └── sample_request.json
├── models/                           # generated artifacts and metrics
├── notebooks/                        # experimental laboratory
├── src/
│   └── ml_project_template/
│       ├── config.py                 # typed settings and paths
│       ├── data.py                   # lineage-aware training-data loading
│       ├── data_contract.py          # versioned schema and deterministic checks
│       ├── example_data.py           # deterministic neutral fixture
│       ├── evaluation.py             # splits, metrics, and diagnostics
│       ├── experiment.py             # MLflow comparison, selection, and gate
│       ├── model_card.py              # generated evidence summary
│       ├── model_package.py           # immutable build and validation contract
│       ├── pipeline.py               # preprocessing + estimator builder
│       ├── prepare_data.py            # idempotent raw-to-prepared CLI
│       ├── lineage.py                 # content hashes and manifest verification
│       ├── tracking.py               # MLflow setup and lineage artifacts
│       ├── train.py                  # authoritative training command
│       └── predict.py                # authoritative inference command
├── pyproject.toml                    # package and dependency groups
└── requirements.txt                  # compatibility installation
```

## Why preprocessing is inside the model artifact

Numeric imputation/scaling and categorical imputation/encoding learn information
from training data. Saving only the estimator would require inference code to
reconstruct those transformations perfectly. Instead, training fits and saves
one sklearn `Pipeline`; prediction loads that same object and accepts raw fields.

This prevents the earlier failure mode in which training and inference used
different preprocessing implementations and different model filenames.

## Using the template for a real project

The Phase 1 example supports tabular binary classification only. When creating a
real project:

1. rename the Python package from `ml_project_template` to the project package;
2. replace the synthetic generator with real ingestion logic;
3. define explicit feature and target columns in the typed settings;
4. choose a split strategy matching time, groups, or other prediction constraints;
5. replace or extend the baseline pipeline through notebook experiments;
6. promote accepted experiments into shared code and tests;
7. remove the generic example after the real end-to-end path works.

Do not present the included logistic regression or synthetic metrics as evidence
for a real use case.

## What is deliberately deferred

Prometheus metrics, Grafana dashboards, alerts, drift detection, registry
promotion, and production deployment remain later extension phases.
