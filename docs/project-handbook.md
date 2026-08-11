# ML Project Template Handbook

## About this handbook

This is the detailed how-to reference for working with `ml-project-template`.
Unlike the README, which provides orientation and a quick start, this handbook
explains how the current system works, how to operate it, what to replace for a
real project, and how to diagnose common failures.

The handbook will grow with the implementation phases. It currently documents
the completed **Phase 1 core**, **Phase 2 quality layer**, **Phase 3
experiment-management layer**, **Phase 4 data layer**, and **Phase 5 model-build
layer**:

- installable Python package;
- deterministic, domain-neutral example data;
- minimum data validation;
- leakage-safe train/test separation;
- one fitted preprocessing-and-estimator pipeline;
- one authoritative model artifact;
- command-line raw inference;
- optional Streamlit and notebook interfaces;
- unit, integration, and repository-contract tests;
- branch coverage, formatting, linting, typing, and security scanning;
- a one-command local quality gate;
- least-privilege GitHub Actions CI on Python 3.10 and 3.12;
- MLflow baseline/tuned comparison, lineage, diagnostics, and signatures;
- development-only model selection and an untouched final test set;
- a packaging quality gate and generated model card;
- an idempotent data-preparation CLI with schema and output validation;
- deterministic profiles and content-addressed lineage manifests;
- random and forward-only temporal split policies;
- path-filtered data CI with generated artifact upload;
- one shared local/CI training-package command;
- validated MLflow Model bundles with commit-SHA identity and retention.

Later editions will add serving, monitoring, deployment, and release-promotion
workflows.

---

## 1. Mental model

The repository has two kinds of code:

1. **Shared code** in `src/ml_project_template/` is the current authoritative
   implementation used by commands and applications.
2. **Experimental code** in `notebooks/` may test alternative ideas. An accepted
   experiment should be promoted into shared code and tested before training or
   inference depends on it.

The Phase 1 execution path is:

```text
data/raw/example.csv
        ↓
contract validation + deterministic preparation
        ↓
data/processed/training.csv + profile.json + manifest.json
        ↓
verify prepared-data SHA-256 before fitting
        ↓
load and validate required columns
        ↓
split raw rows into train and test
        ↓
fit preprocessing on training rows only
        ↓
fit logistic regression on transformed training rows
        ↓
evaluate on transformed test rows
        ↓
models/model.joblib + models/metrics.json
        ↓
raw JSON record → saved pipeline → prediction
```

The critical rule is that `models/model.joblib` contains both preprocessing and
the estimator. Prediction does not recreate preprocessing separately.

---

## 2. Repository orientation

```text
ml-project-template/
├── app/
│   └── main.py
├── data/
│   ├── external/
│   ├── interim/
│   ├── processed/
│   └── raw/
├── docs/
│   ├── current-state-assessment.md
│   ├── project-extension-plan.md
│   └── project-handbook.md
├── examples/
│   └── binary_classification/
│       └── sample_request.json
├── models/
├── notebooks/
│   ├── 01_eda_and_discovery.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_inference_test.ipynb
├── src/
│   └── ml_project_template/
│       ├── __init__.py
│       ├── config.py
│       ├── data.py
│       ├── example_data.py
│       ├── pipeline.py
│       ├── predict.py
│       └── train.py
├── .gitignore
├── pyproject.toml
├── README.md
└── requirements.txt
```

### Important files

| File | Purpose |
|---|---|
| `pyproject.toml` | Package metadata, dependencies, optional groups, and installed commands |
| `config.py` | Typed settings, feature names, target, split parameters, and paths |
| `data_contract.py` | Versioned schema, safe canonicalization, and deterministic profile |
| `prepare_data.py` | Idempotent raw-to-prepared command and evidence writer |
| `lineage.py` | SHA-256 identities and manifest verification |
| `data.py` | Lineage-aware training-data loading and contract validation |
| `example_data.py` | Reproducible synthetic dataset generator |
| `pipeline.py` | Preprocessing plus baseline estimator construction |
| `train.py` | Authoritative training and artifact-writing command |
| `model_package.py` | Immutable model-bundle build and independent validation |
| `predict.py` | Authoritative model loading and raw-record prediction |
| `sample_request.json` | Known-valid inference request, including an unseen category |
| `app/main.py` | Optional Streamlit client of the prediction function |

---

## 3. Environment setup

### 3.1 Requirements

- Python 3.10 or newer
- PowerShell commands shown below, or their shell equivalent
- internet access during the first dependency installation

Check Python:

```powershell
python --version
```

### 3.2 Create an isolated environment

Run commands from the repository root:

```powershell
cd D:\projects\ml-project-template
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

The prompt normally shows `(.venv)` after activation.

If PowerShell blocks activation, either adjust the execution policy according to
your machine policy or call the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe --version
```

### 3.3 Install the package

Install the minimum core:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

`-e` means editable installation. Imports and commands point at the current
source tree, so source edits are available without reinstalling after every
change.

Verify the installation:

```powershell
python -c "import ml_project_template; print(ml_project_template.__version__)"
python -m pip check
```

Expected package version:

```text
0.1.0
```

### 3.4 Optional dependency groups

Core training and prediction:

```powershell
python -m pip install -e .
```

Notebook and experiment tools:

```powershell
python -m pip install -e ".[experiment]"
```

Streamlit application:

```powershell
python -m pip install -e ".[app]"
```

Phase 2 development and quality tools:

```powershell
python -m pip install -e ".[dev]"
```

Multiple groups can be installed together:

```powershell
python -m pip install -e ".[experiment,app,dev]"
```

`requirements.txt` is retained as a compatibility shortcut for the notebook and
application experience:

```powershell
python -m pip install -r requirements.txt
```

Prefer `pyproject.toml` groups in new automation because they express why each
dependency is needed.

---

## 4. Generate the example dataset

The template does not commit a real-world dataset. Instead, it generates a
deterministic fixture with neutral names.

Run:

```powershell
python -m ml_project_template.example_data
```

Default output:

```text
data/raw/example.csv
```

Default behavior:

- 400 rows;
- fixed random seed `42`;
- two numeric features;
- two categorical features;
- one binary target;
- some missing numeric and categorical values;
- a learnable but noisy synthetic relationship.

The missing values deliberately prove that the fitted imputers are used.

### 4.1 Customize generation

Generate a different number of rows:

```powershell
python -m ml_project_template.example_data --rows 1000
```

Change the seed:

```powershell
python -m ml_project_template.example_data --random-state 7
```

Write elsewhere:

```powershell
python -m ml_project_template.example_data --output artifacts/manual-check/example.csv
```

Show all generator options:

```powershell
python -m ml_project_template.example_data --help
```

### 4.2 Inspect the generated data

```powershell
python -c "import pandas as pd; data = pd.read_csv('data/raw/example.csv'); print(data.head()); print(data.shape); print(data.isna().sum())"
```

Expected columns:

```text
numeric_feature_1
numeric_feature_2
category_feature_1
category_feature_2
target
```

The names are intentionally non-domain-specific. The example exists to test the
template's connections, not to suggest a business problem.

---

## 5. Prepare data and understand the Phase 4 contract

Run deterministic preparation after generation or real ingestion:

```powershell
python -m ml_project_template.prepare_data
```

Default outputs:

```text
data/processed/training.csv
data/processed/profile.json
data/processed/manifest.json
```

The versioned contract is defined by `Settings`, enforced by
`data_contract.py`, and rechecked by `data.py` before training.

Training currently requires:

- at least the configured minimum number of rows;
- exactly the declared columns unless extras are explicitly allowed;
- parseable, finite non-missing numeric values while permitting configured
  feature nulls for the fitted imputer;
- non-blank categorical values within the configured cardinality limit;
- no duplicate rows or column names;
- the configured target column;
- no missing target values;
- both binary target values `0` and `1`, with enough class support for splitting
  and cross-validation;
- a valid non-missing UTC-normalizable time column when temporal splitting is
  enabled.

Missing feature values are permitted because the pipeline imputes them.
Preparation does not impute, encode, scale, select features, or fit any learned
statistics. Those operations remain inside the sklearn pipeline after the split.

### 5.1 Preparation options

```powershell
python -m ml_project_template.prepare_data `
  --input path/to/raw.csv `
  --output data/processed/training.csv `
  --manifest data/processed/manifest.json `
  --profile data/processed/profile.json `
  --source-uri "s3://bucket/data.csv?versionId=immutable-version"
```

`--source-uri` is optional but should identify an immutable upstream object
version, snapshot, or dataset release when one exists. The SHA-256 reference is
always generated regardless.

### 5.2 Idempotency and tamper detection

With identical input bytes, configuration, and code revision, repeated commands
write identical prepared CSV, profile, and manifest bytes. Writes are atomic so
an interrupted replacement does not expose a partially written file.

Before fitting, `train` and `experiment` recompute the prepared CSV hash and
compare it with `manifest.json`. Modified bytes stop the workflow before a model
or MLflow run is created.

The Phase 1 contract does not yet validate:

- detailed pandas or semantic data types;
- numeric bounds;
- allowed categorical domains;
- identifier uniqueness;
- duplicate policy;
- sensitive columns;
- schema drift;
- temporal or group leakage.

Those belong to the richer data-contract phase. For a real project, do not assume
the Phase 1 checks are sufficient.

---

## 6. Configure the baseline

Open:

```text
src/ml_project_template/config.py
```

The typed `Settings` object currently defines:

```python
target_column = "target"
numeric_features = (
    "numeric_feature_1",
    "numeric_feature_2",
)
categorical_features = (
    "category_feature_1",
    "category_feature_2",
)
test_size = 0.2
random_state = 42
```

It also derives these authoritative paths:

```text
data/raw/example.csv
models/model.joblib
models/metrics.json
```

### 6.1 Override the project root

By default, the root is derived from the installed source file. It can be
overridden with `ML_PROJECT_ROOT`:

```powershell
$env:ML_PROJECT_ROOT = "D:\path\to\project"
python -m ml_project_template.train
```

Remove the temporary override from the current shell:

```powershell
Remove-Item Env:ML_PROJECT_ROOT
```

### 6.2 Validate custom settings in Python

```python
from pathlib import Path

from ml_project_template.config import Settings

settings = Settings(
    project_root=Path.cwd(),
    target_column="target",
    numeric_features=("numeric_feature_1", "numeric_feature_2"),
    categorical_features=("category_feature_1", "category_feature_2"),
    test_size=0.2,
    random_state=42,
)
```

Invalid split ratios, empty target names, empty feature sets, and duplicate
feature names fail during settings construction.

Configuration files and environment profiles are intentionally deferred until
the configuration requirements are better understood.

---

## 7. Understand the model pipeline

`pipeline.py` constructs one sklearn `Pipeline`:

```text
Pipeline
├── preprocessor: ColumnTransformer
│   ├── numeric
│   │   ├── SimpleImputer(strategy="median")
│   │   └── StandardScaler()
│   └── categorical
│       ├── SimpleImputer(strategy="most_frequent")
│       └── OneHotEncoder(handle_unknown="ignore")
└── estimator
    └── LogisticRegression(max_iter=1000, random_state=42)
```

### Why split before fitting the pipeline?

Imputation medians, category vocabularies, and scaling statistics are learned
from data. The training function splits raw rows first and then calls:

```python
model.fit(X_train, y_train)
```

The test rows are only passed to the already-fitted pipeline during evaluation.
This prevents test statistics from influencing preprocessing.

### Why save the complete pipeline?

If preprocessing and the estimator were saved separately, inference would need
to coordinate two files and exactly reproduce feature order. A complete pipeline
accepts the raw configured columns and owns the full transformation sequence.

### Why use a simple logistic regression?

The initial shared model is a stable control. It proves architecture and data
flow before a notebook discovers a better hypothesis. It is not intended to win
a model comparison.

---

## 8. Train the example model

Generate and prepare data first, then run:

```powershell
python -m ml_project_template.example_data
python -m ml_project_template.prepare_data
python -m ml_project_template.train
```

Training performs these steps:

1. verifies the prepared CSV against its manifest hash;
2. reads and revalidates the complete data contract;
3. selects configured features and target;
4. creates the configured stratified-random or chronological train/test split;
5. constructs a new unfitted pipeline;
6. fits the entire pipeline on training rows;
7. predicts and calculates metrics on test rows;
8. saves the fitted pipeline;
9. saves machine-readable metrics;
10. writes concise key-value logs and prints metrics as JSON.

### 8.1 Training outputs

```text
models/model.joblib
models/metrics.json
```

Both files are generated and ignored by Git.

`model.joblib` is authoritative for Phase 1 inference. `metrics.json` contains:

- accuracy;
- precision for class `1`;
- recall for class `1`;
- F1 for class `1`;
- ROC AUC;
- training-row count;
- test-row count.

### 8.2 Use custom paths

```powershell
python -m ml_project_template.train `
  --data artifacts/manual-check/example.csv `
  --artifact artifacts/manual-check/model.joblib `
  --metrics artifacts/manual-check/metrics.json
```

Show all training options:

```powershell
python -m ml_project_template.train --help
```

### 8.3 Inspect the artifact

```powershell
python -c "import joblib; model = joblib.load('models/model.joblib'); print(model.named_steps); print(model.feature_names_in_)"
```

Expected pipeline step names:

```text
preprocessor
estimator
```

### 8.4 Interpret the example metrics correctly

The synthetic generator creates the target from a noisy relationship involving
the four example features. Good metrics only demonstrate that the workflow can
learn that artificial relationship. They are not evidence about a real domain,
fairness, safety, production quality, or business value.

---

## 9. Run inference

Use the committed request file:

```powershell
python -m ml_project_template.predict `
  --input-file examples/binary_classification/sample_request.json
```

Expected response structure:

```json
{
  "prediction": 1,
  "probability": 0.79
}
```

The exact probability is deterministic for the default data and package
versions but should not be treated as a permanent API snapshot yet.

### 9.1 Why the sample uses `unseen_category`

`unseen_category` is deliberately absent from training. Successful prediction
shows that the fitted `OneHotEncoder(handle_unknown="ignore")` inside the model
artifact is being used during inference.

### 9.2 Use inline JSON

The CLI also accepts an inline object:

```powershell
python -m ml_project_template.predict `
  --input-json '{"numeric_feature_1":0.25,"numeric_feature_2":-0.5,"category_feature_1":"category_a","category_feature_2":"group_1"}'
```

Shell quoting differs between PowerShell, Command Prompt, Bash, and CI runners.
`--input-file` is the most portable manual test.

### 9.3 Call prediction from Python

```python
from ml_project_template.config import get_settings
from ml_project_template.predict import predict_record

settings = get_settings()

record = {
    "numeric_feature_1": 0.25,
    "numeric_feature_2": -0.5,
    "category_feature_1": "category_a",
    "category_feature_2": "group_1",
}

result = predict_record(record, settings.model_artifact_path)
print(result)
```

### 9.4 Input behavior

- Every configured feature must be present.
- Extra keys are currently ignored when the dataframe is ordered to the fitted
  feature columns.
- Unknown categories are accepted.
- Values that cannot be processed by the configured pipeline fail.
- `probability` is `null` only if the estimator does not expose a usable positive
  class probability.

The serving phase will define a stricter versioned request and response schema.

---

## 10. Run the optional Streamlit application

Install the application group:

```powershell
python -m pip install -e ".[app]"
```

Generate data and train if needed:

```powershell
python -m ml_project_template.example_data
python -m ml_project_template.train
```

Start Streamlit:

```powershell
streamlit run app/main.py
```

The UI provides the four generic input fields and calls `predict_record`. It does
not contain a second preprocessing implementation.

The application is a demonstration client, not a production API. Closing the
terminal stops it.

---

## 11. Work with the notebooks

Install experiment dependencies:

```powershell
python -m pip install -e ".[experiment]"
jupyter lab
```

The notebooks have different roles:

| Notebook | Intended use |
|---|---|
| `01_eda_and_discovery.ipynb` | Explore only the permitted training data and form hypotheses |
| `02_preprocessing.ipynb` | Experiment with alternative preprocessing ideas |
| `03_model_training.ipynb` | Experiment with model comparison and MLflow ideas |
| `04_inference_test.ipynb` | Exercise the authoritative shared prediction implementation |

The first three remain laboratories and will be integrated more deeply during
later phases. The inference notebook already imports `predict_record` rather
than duplicating artifact loading and preprocessing.

### Notebook promotion rule

```text
shared baseline
      ↓
notebook-local hypothesis
      ↓
consistent comparison
      ↓
rejected → retain learning notes if useful
      │
      └── accepted → refactor into shared code → test → use in training/inference
```

Notebook duplication is acceptable while an idea is explicitly experimental.
It becomes dangerous when notebook results use one implementation while the
saved or served model uses another without that difference being declared.

---

## 12. Use the installed commands

Editable installation also creates command aliases:

```powershell
ml-template-generate-data
ml-template-train
ml-template-predict --input-file examples/binary_classification/sample_request.json
```

The `python -m ...` versions are used throughout this handbook because they make
the active Python interpreter explicit and are easier to diagnose when multiple
environments exist.

---

## 13. Turn the template into a real project

Do not begin by replacing only the CSV. Work through the contract deliberately.

### Step 1: Rename project identities

- rename the repository;
- rename `src/ml_project_template/` to a valid Python package name using
  underscores;
- change the project name and installed command names in `pyproject.toml`;
- update imports, README text, and application labels.

Example:

```text
repository: customer-behavior-model
package:    customer_behavior_model
```

### Step 2: Replace synthetic ingestion

- define where raw data comes from;
- keep source data immutable;
- remove the example generator when the real ingestion path is proven;
- never commit private data merely to make the template run.

### Step 3: Define the real contract

- target name and meaning;
- feature names and roles;
- required and optional values;
- types and ranges;
- missing-value policy;
- identifiers and duplicates;
- sensitive fields;
- prediction-time availability.

### Step 4: Choose the correct split

Use random stratification only when rows are independent and exchangeable. Use a
time-based or group-aware strategy when prediction conditions require it.

### Step 5: Establish the real baseline

Replace the generic transformations and logistic regression only as required by
the problem. Keep the baseline simple enough to understand and reproduce.

### Step 6: Experiment in notebooks

Try alternate features, preprocessing, models, thresholds, and metrics without
immediately turning them into shared behavior.

### Step 7: Promote accepted behavior

Move successful ideas into shared code, then add tests before treating the new
pipeline as authoritative.

### Step 8: Remove the generic example

Delete the example-specific files only after the real generate/load → train →
predict path works. Keep a small synthetic fixture for automated tests.

---

## 14. Troubleshooting

### `ModuleNotFoundError: No module named 'ml_project_template'`

Cause: the package is not installed in the active interpreter, or the wrong
virtual environment is active.

Check:

```powershell
python -c "import sys; print(sys.executable)"
python -m pip show ml-project-template
```

Fix:

```powershell
python -m pip install -e .
```

Do not repair this with `sys.path.append(...)`.

### `Training data not found`

Generate the default data:

```powershell
python -m ml_project_template.example_data
python -m ml_project_template.prepare_data
```

Or prepare the intended CSV explicitly before training:

```powershell
python -m ml_project_template.prepare_data --input path/to/raw.csv
python -m ml_project_template.train
```

### `Prepared data hash does not match its manifest`

The prepared CSV changed after its manifest was written, or the wrong manifest
was paired with it. Do not edit `training.csv` manually. Rerun preparation from
the intended immutable raw input and inspect the changed hashes.

### `Training data is missing columns`

The CSV does not match the feature and target names in `Settings`. Inspect both:

```powershell
python -c "import pandas as pd; print(pd.read_csv('data/raw/example.csv').columns.tolist())"
python -c "from ml_project_template.config import get_settings; print(get_settings().feature_columns)"
```

Update the real project's settings or ingestion logic; do not silently discard
the validation error.

### `The Phase 1 example expects a binary target`

The current shared baseline supports only binary labels `0` and `1`, and both
must appear. Multiclass and regression require explicit task adapters in later
work rather than bypassing the check.

### `Model artifact not found`

Run training first:

```powershell
python -m ml_project_template.train
```

If a custom artifact path was used during training, pass the same path during
prediction:

```powershell
python -m ml_project_template.predict `
  --artifact artifacts/manual-check/model.joblib `
  --input-file examples/binary_classification/sample_request.json
```

### Prediction input is missing columns

Compare the request with the expected features:

```powershell
python -c "import joblib; model = joblib.load('models/model.joblib'); print(model.feature_names_in_.tolist())"
```

All listed fields must be present, even if a field's value is missing and handled
by an imputer.

### Inline JSON cannot be parsed

Use a JSON file to avoid shell-quoting differences:

```powershell
python -m ml_project_template.predict `
  --input-file examples/binary_classification/sample_request.json
```

### Editable installation reports a missing build tool

Upgrade the packaging tools and retry:

```powershell
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

### Streamlit command is unavailable

Install the application dependency group:

```powershell
python -m pip install -e ".[app]"
```

---

## 15. Clean generated outputs safely

The following outputs are reproducible and ignored:

```text
data/raw/example.csv
models/model.joblib
models/metrics.json
artifacts/
```

Normally, regenerate them rather than committing them. Do not recursively delete
the entire `data/` or repository root. Preserve `.gitkeep` placeholders and any
real data that belongs to your project.

The virtual environment is also local and ignored:

```text
.venv/
```

Deleting it removes installed packages for this project but not source code. It
can be recreated from `pyproject.toml`.

---

## 16. Phase 1 verification checklist

Use this checklist when validating a fresh clone or a generated project:

- [ ] Python 3.10+ is active.
- [ ] `python -m pip install -e .` succeeds.
- [ ] `python -m pip check` reports no broken requirements.
- [ ] `python -m ml_project_template.example_data` creates the CSV.
- [ ] The CSV contains the four configured features and binary target.
- [ ] `python -m ml_project_template.prepare_data` creates canonical data,
  profile, and manifest files.
- [ ] Repeating preparation preserves all three output hashes.
- [ ] `python -m ml_project_template.train` succeeds.
- [ ] `models/model.joblib` exists.
- [ ] `models/metrics.json` exists.
- [ ] The saved object has `preprocessor` and `estimator` steps.
- [ ] Prediction succeeds using `sample_request.json`.
- [ ] The unseen category does not crash prediction.
- [ ] Removing a required request field produces a clear validation failure.
- [ ] Generated datasets, models, environments, and caches do not appear as
  unignored Git changes.

Inspect Git state:

```powershell
git status --short --ignored
```

---

## 17. Current limitations and next expansion

Phases 1 through 5 intentionally do not yet provide:

- domain-specific semantic, fairness, privacy, or sensitive-field validation;
- durable remote storage for non-regenerable datasets;
- a long-lived model registry or promotion environment;
- prediction API or container;
- monitoring, dashboards, alerts, or drift detection.

Phase 6 should serve the exact Phase 5 MLflow Model through a typed API and
non-root container with health checks and contract smoke tests.

---

## 18. Phase 2 testing and quality reference

### 18.1 Install the quality tools

```powershell
python -m pip install -e ".[dev]"
```

This group installs:

| Tool | Responsibility |
|---|---|
| Ruff formatter | enforces one Python formatting style |
| Ruff linter | detects import, correctness, modernization, bug-risk, and simplification issues |
| mypy | checks typed shared-source interfaces |
| Bandit | scans source for common Python security mistakes |
| pytest | executes behavioral contracts |
| pytest-cov/coverage | reports statement and branch coverage |

These tools are development dependencies. Core users do not install them through
`pip install -e .`.

### 18.2 Run the complete local gate

On Windows:

```powershell
PowerShell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

`-ExecutionPolicy Bypass` applies only to this child PowerShell process. It is
included because some Windows installations block local `.ps1` files by default.
It does not permanently change the user's execution policy.

If the active environment already permits scripts:

```powershell
.\scripts\check.ps1
```

The script stops at the first failed stage and runs:

```powershell
python -m ruff format --check .
python -m ruff check .
python -m mypy src
python -m bandit -q -r src app
python -m pytest
```

### 18.3 Run individual checks

Check formatting without changing files:

```powershell
python -m ruff format --check .
```

Format Python files:

```powershell
python -m ruff format .
```

Run the linter:

```powershell
python -m ruff check .
```

Apply only Ruff's safe automatic fixes, then inspect the diff:

```powershell
python -m ruff check --fix .
git diff
```

Run source type checking:

```powershell
python -m mypy src
```

Run the security scan:

```powershell
python -m bandit -q -r src app
```

Run all tests and coverage:

```powershell
python -m pytest
```

Run one test file:

```powershell
python -m pytest tests/unit/test_data.py
```

Run one test by name:

```powershell
python -m pytest tests/integration/test_training_inference.py::test_training_artifact_is_compatible_with_raw_inference
```

Show captured logs/output while testing:

```powershell
python -m pytest -s
```

### 18.4 Test organization

```text
tests/
├── conftest.py
├── unit/
│   ├── test_config.py
│   ├── test_data.py
│   ├── test_example_data.py
│   ├── test_pipeline.py
│   └── test_predict.py
├── integration/
│   ├── test_cli.py
│   └── test_training_inference.py
└── repository/
    └── test_repository_contract.py
```

The categories have different purposes:

- **Unit tests** isolate configuration, data-contract, generator, pipeline, and
  prediction behavior.
- **Integration tests** generate data, train, save, load, and predict through the
  real public interfaces and CLI entry points.
- **Repository-contract tests** verify template-level promises such as valid
  notebook JSON, exact sample-request fields, and absence of runtime path hacks.

All test data is generated under pytest temporary directories. Tests do not need
`data/raw/example.csv`, `models/model.joblib`, private data, network access, or
repository secrets.

### 18.5 Behaviors protected by the suite

The current 76 tests protect:

- authoritative path resolution;
- environment-based project-root configuration;
- invalid split ratios, empty target names, empty feature sets, and duplicate
  features;
- deterministic synthetic generation;
- the exact neutral example schema and balanced target;
- empty datasets, missing columns, missing targets, invalid target classes, and
  missing files;
- raw fitting with missing values;
- ordered configured features;
- unseen categorical values;
- missing model artifacts and prediction fields;
- exact prediction response keys;
- legitimate probability `0.0` values;
- estimators without probability support;
- file-based and inline-JSON CLI execution;
- one complete generate → train → save → load → raw predict lifecycle;
- valid notebook JSON and absence of legacy `sys.path` mutation.
- MLflow run comparison, selection metadata, model signatures, lineage
  artifacts, model-card generation, and failed-gate non-packaging.
- deterministic preparation, schema failures, content hashes, tamper detection,
  manifest-aware fitting, and chronological holdout behavior.
- immutable package naming, complete inventories, file integrity, dual-format
  inference, required MLflow metadata, and failed-build cleanup.

### 18.6 Coverage policy

Pytest is configured to measure both statements and branches in
`ml_project_template`. The gate requires at least 90% total coverage. The current
suite remains at or above the required 90% branch-coverage threshold.

Coverage is evidence that relevant paths were executed, not proof that assertions
are meaningful. A new test should protect a behavior or failure mode; do not add
empty tests merely to increase the number.

Coverage output identifies missed lines and partial branches. HTML output can be
generated when deeper inspection is helpful:

```powershell
python -m pytest --cov-report=html
```

Then open `htmlcov/index.html` locally. `htmlcov/` is ignored.

### 18.7 Add a new behavior safely

Use this sequence:

1. decide whether the change belongs in shared code or remains a notebook
   experiment;
2. describe its input, output, and expected failures;
3. add or update a focused unit test;
4. add an integration test if the artifact or lifecycle contract changes;
5. implement the behavior;
6. run the individual test while iterating;
7. run the complete quality gate;
8. inspect the diff before committing.

Bug fixes should normally begin with a failing regression test that demonstrates
the bug. The test should pass only after the fix is implemented.

### 18.8 Interpret common quality failures

`ruff format --check` failure:

- one or more Python files differ from the configured formatter;
- run `python -m ruff format .`, inspect the diff, and rerun the gate.

`ruff check` failure:

- read the rule code and explanation;
- use `--fix` only when the proposed transformation preserves intent;
- avoid blanket rule suppression without documenting why.

`mypy` failure:

- an annotated shared-code contract is inconsistent;
- prefer correcting the type or control flow over adding `Any` or ignores;
- third-party missing stubs are currently ignored, but project code is checked.

`bandit` failure:

- inspect the confidence, severity, and code path;
- repair real risks;
- suppress a false positive only beside the relevant line with a concise reason.

`pytest` assertion failure:

- the behavior differs from its executable contract;
- determine whether the implementation regressed or the intended contract
  deliberately changed;
- update a test only when the desired behavior truly changed.

Coverage-threshold failure:

- all tests may pass while important new branches remain unexercised;
- add behavior-focused tests for those paths rather than lowering the threshold
  automatically.

### 18.9 GitHub Actions CI

`.github/workflows/ci.yml` runs for:

- pushes;
- pull requests;
- manual workflow dispatch.

It tests Python 3.10 and 3.12 on Ubuntu and performs:

1. clean checkout;
2. Python setup with pip caching;
3. editable installation with the `dev,experiment` groups;
4. dependency consistency check;
5. format check;
6. lint;
7. type check;
8. Bandit scan;
9. pytest with branch coverage.

Security and resource decisions:

- workflow permissions are limited to `contents: read`;
- no secrets are required;
- duplicate runs on the same ref are cancelled;
- each matrix job has a 15-minute timeout;
- tests generate their own data and artifacts in temporary directories;
- pull requests do not publish packages, models, or containers.

Local success proves the commands work on this Windows/Python 3.10 environment.
The remote GitHub status is authoritative for Ubuntu and Python 3.12 after the
commit is pushed.

### 18.10 Phase 2 completion check

The expected local result is:

```text
Ruff format: pass
Ruff lint: pass
mypy: pass
Bandit: pass
pytest: 76 passed
branch coverage: at least 90%
```

If test count changes later, the behavioral coverage and passing quality gate
matter more than preserving the number `33`.

---

## 19. Phase 3 experiment management and evaluation

### 19.1 What this layer is for

The Phase 1 `train` command remains the smallest raw-data-to-model example. The
Phase 3 `experiment` command answers a larger question: **which candidate should
be packaged, and what evidence supports that decision?**

Its data flow is deliberately asymmetric:

```text
raw data
  -> reserve untouched test rows
  -> compare baseline and tuned candidates by development-only CV
  -> select the highest mean development CV F1
  -> refit that candidate on all development rows
  -> evaluate that winner once on the untouched test rows
  -> package only when test F1 passes the configured gate
```

The test set is not another tuning fold. Looking at it while choosing candidates
would make the reported final result part of the optimization process.

### 19.2 Install and run

Install the experiment layer and generate the neutral fixture:

```powershell
python -m pip install -e ".[experiment]"
python -m ml_project_template.example_data
python -m ml_project_template.prepare_data
python -m ml_project_template.experiment
```

The installed command is equivalent:

```powershell
ml-template-experiment
```

Useful overrides:

```powershell
python -m ml_project_template.experiment `
  --data data/raw/example.csv `
  --artifact models/model.joblib `
  --metrics models/metrics.json `
  --model-card models/model-card.md `
  --experiment-name my-local-comparison `
  --min-test-f1 0.75
```

The default gate is test F1 >= `0.70`. A threshold must be between zero and one.
For a real project, choose it from domain costs and baseline evidence rather
than retaining the template number blindly.

### 19.3 What gets compared

Both candidates use the authoritative pipeline builder, so their preprocessing
contract is identical:

- `baseline`: default logistic-regression settings;
- `tuned`: `GridSearchCV` over `C` values `0.1`, `1.0`, and `10.0`, with and
  without balanced class weights;
- both use stratified shuffled CV with the configured seed;
- selection currently supports mean CV F1 only.

This is a small teaching search, not a claim that logistic regression or this
grid is suitable for every dataset. Add a candidate only after defining its
valid parameter space and evaluation policy.

### 19.4 MLflow runs and evidence

The default tracking URI is the ignored local SQLite database `mlflow.db`.
Each experiment execution creates two MLflow runs. Both record candidate
parameters, cross-validation metrics, selection tags, and reproducibility
evidence. The tuned run also stores its complete CV results.

The selected run additionally records:

- untouched-test accuracy, precision, recall, F1, and ROC AUC;
- classification report and confusion matrix;
- candidate scores and the explicit statement that test data was not used for
  selection;
- quality-gate result;
- the complete fitted sklearn pipeline;
- inferred model signature and a five-row input example;
- generated model card.

Reproducibility evidence includes the verified prepared-data reference, raw
input reference, optional immutable source URI, exact prepared-file SHA-256,
current Git revision when available, resolved settings, and dataset profile. A
hash proves identity; the source URI or artifact store must preserve availability.

### 19.5 Inspect runs locally

From this repository's current path on Windows:

```powershell
mlflow ui --backend-store-uri sqlite:///D:/projects/ml-project-template/mlflow.db
```

Open `http://127.0.0.1:5000`, choose the `ml-project-template` experiment, and
compare the baseline and tuned runs. If the project is moved, update the
absolute path in the SQLite URI.

SQLite is the tracking database; it is not by itself a model registry,
promotion policy, or immutable release system.

### 19.6 Use a remote tracking server

Set `MLFLOW_TRACKING_URI` before running the same command:

```powershell
$env:MLFLOW_TRACKING_URI = "https://mlflow.example.internal"
python -m ml_project_template.experiment
```

Remove the shell-local override afterward:

```powershell
Remove-Item Env:MLFLOW_TRACKING_URI
```

Authentication and TLS depend on the chosen server. Keep tokens and credentials
in environment variables or a secret manager, never in repository files.

### 19.7 Packaging outcomes

On success, local ignored outputs are:

```text
mlflow.db
models/model.joblib
models/metrics.json
models/model-card.md
```

`metrics.json` connects the candidate run IDs, selection scores, final metrics,
data/code identity, and MLflow model URI. `model.joblib` remains compatible with
the Phase 1 raw-record prediction command.

When the gate fails, the selected run still retains the failure metrics and
diagnostics so the result can be investigated, but the command raises
`QualityGateError` before writing the local model, metrics file, model card, or
MLflow model. A failed experiment is evidence, not a deployable artifact.

### 19.8 Model-card workflow

`docs/model-card-template.md` is the human checklist for real projects. A
successful experiment generates `models/model-card.md` with the facts available
automatically. Replace or expand its generic intended-use and limitation text
when adapting the template. Do not infer fairness, calibration, robustness, or
business value from the synthetic fixture's metrics.

### 19.9 Verify Phase 3

Run the full automated gate:

```powershell
PowerShell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

The Phase 3 integration tests verify that:

- exactly two distinct runs are created;
- selection is based on development CV evidence;
- the selected run carries final-test metrics and packaging metadata;
- the logged model has a signature and accepts raw mixed-type input;
- a deliberately failed gate produces no deployable artifacts.

For a manual smoke test, run the experiment, inspect both runs in the UI, then
use the packaged local artifact:

```powershell
python -m ml_project_template.predict `
  --input-file examples/binary_classification/sample_request.json
```

---

## 20. Phase 4 data automation and versioning

### 20.1 Data lifecycle

Phase 4 separates two meanings often hidden behind the word preprocessing:

```text
raw source
  -> deterministic preparation (safe before splitting)
  -> prepared CSV + profile + manifest
  -> verify content hash
  -> reserve development/test boundary
  -> fit learned preprocessing inside the model pipeline on development only
```

Deterministic preparation may enforce types, trim categorical whitespace,
canonicalize timestamps, order declared columns, and validate constraints. It
must not learn medians, category vocabularies, scaling statistics, clipping
bounds, feature selection, or target-dependent transformations from all rows.

### 20.2 Contract settings

The generic contract exposes these Phase 4 settings:

| Setting | Default | Meaning |
|---|---:|---|
| `data_contract_version` | `1` | version of prepared-data semantics |
| `minimum_rows` | `20` | smallest accepted dataset |
| `max_category_cardinality` | `100` | per-categorical-feature safety limit |
| `allow_extra_columns` | `False` | reject undeclared columns by default |
| `split_strategy` | `random` | `random` or `time` |
| `time_column` | `None` | required separate timestamp for `time` |

For a real project, extend the contract with domain ranges, permitted category
sets, uniqueness rules, sensitive-field exclusions, and feature-availability
checks. A generic template cannot infer those safely.

### 20.3 Prepared artifacts

`training.csv` is the canonical declared dataset. `profile.json` records row and
column counts, order, dtypes, null counts, numeric summaries, categorical
cardinality, and target distribution. `manifest.json` records:

- contract snapshot and version;
- preparation implementation and Git revision;
- raw input path, SHA-256, immutable hash reference, and optional source URI;
- prepared output path, SHA-256, and immutable hash reference;
- profile path and SHA-256.

The manifest contains no wall-clock timestamp, so the same input, settings, and
code revision reproduce identical evidence bytes.

### 20.4 Real source references

For local or generated data, `sha256:<digest>` is the immutable identity. For
remote data, also pass a source URI that cannot silently change, for example:

- an object-store URI with a version ID;
- a warehouse snapshot/table version;
- a dataset release ID;
- a DVC content revision.

Avoid a mutable reference such as `s3://bucket/latest.csv` by itself. A hash can
detect a change, but it cannot recover bytes that were never retained.

### 20.5 Why DVC is not installed by default

The included dataset is tiny, deterministic, and regenerable. Adding DVC and a
remote solely for this fixture would create configuration without solving a
real storage problem. The default therefore uses Git for code/configuration,
SHA manifests for identity, and GitHub Actions artifacts for short-lived CI
evidence.

Add DVC or versioned object storage when one or more are true:

- datasets are too large for normal artifacts;
- raw data cannot be regenerated;
- multiple people or jobs must retrieve exact versions;
- retention, access control, audit, or rollback requirements exceed CI artifact
  lifetimes;
- dataset versions must be promoted independently from code.

Whichever system is chosen, place its immutable URI/version in `--source-uri`
and preserve the manifest beside model evidence.

### 20.6 Random versus temporal splitting

The default stratified random strategy is appropriate only when rows are
independent and exchangeable. Configure time-aware evaluation in project code:

```python
from pathlib import Path

from ml_project_template.config import Settings

settings = Settings(
    project_root=Path.cwd(),
    split_strategy="time",
    time_column="event_time",
)
```

Preparation requires parseable, non-missing timestamps and canonicalizes them
to UTC. Evaluation sorts stably by time, reserves the newest fraction as final
test data, and uses forward-only `TimeSeriesSplit` during experiment selection.
Both development and final test partitions must contain both target classes.

Do not use temporal splitting merely because a timestamp exists. Use it when
the real prediction task trains on history and predicts future observations.
Group-aware splitting is still a project-specific extension when entities must
not cross partitions.

### 20.7 Data CI workflow

`.github/workflows/data-pipeline.yml` runs manually and when the preparation
contract or implementation changes. It:

1. installs only the core package;
2. generates the deterministic neutral raw fixture;
3. runs the same `prepare_data` module used locally;
4. verifies the manifest hash;
5. uploads `training.csv`, `profile.json`, and `manifest.json` as
   `prepared-data-<commit SHA>` for 14 days.

Generated data is uploaded, not committed. For a private real source, replace
the fixture-generation step with authorized ingestion and review fork/secret
security before enabling pull-request access.

### 20.8 Verify Phase 4 locally

Run preparation twice and compare hashes:

```powershell
python -m ml_project_template.example_data
python -m ml_project_template.prepare_data
Get-FileHash data/processed/training.csv -Algorithm SHA256
Get-FileHash data/processed/profile.json -Algorithm SHA256
Get-FileHash data/processed/manifest.json -Algorithm SHA256
python -m ml_project_template.prepare_data
Get-FileHash data/processed/training.csv -Algorithm SHA256
Get-FileHash data/processed/profile.json -Algorithm SHA256
Get-FileHash data/processed/manifest.json -Algorithm SHA256
```

Each before/after pair should match. Then run training or the Phase 3 experiment
and inspect `data_reference` in `models/metrics.json` and the MLflow run.

Run every automated contract:

```powershell
PowerShell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

The suite covers malformed schema/value cases, repeatability, manifest tampering,
failure before fitting, random and chronological holdouts, MLflow lineage, and
the complete raw-to-prediction workflow.

---

## 21. Phase 5 reproducible training and packaging

### 21.1 What Phase 5 guarantees

Phase 4 proves which data entered the process. Phase 5 proves which complete
model bundle came out and whether another machine can consume it:

```text
verified prepared data + manifest + code/config
  -> shared experiment/quality-gate entry point
  -> selected fitted pipeline
  -> MLflow Model + consumer/evidence files
  -> inventory, hash, metadata, and inference validation
  -> atomic artifacts/model-packages/<version>/ publication
```

The final version directory appears only after every check passes. A failure
during training, quality gating, assembly, hashing, or smoke inference leaves no
partially published package.

### 21.2 Build locally

Install the experiment layer, prepare data, and build:

```powershell
python -m pip install -e ".[experiment]"
python -m ml_project_template.example_data
python -m ml_project_template.prepare_data
python -m ml_project_template.model_package --version local-check
```

Equivalent installed command:

```powershell
ml-template-package-model --version local-check
```

Without `--version`, the builder uses `GITHUB_SHA`, then the current Git commit,
then `local` when no repository identity exists. Versions permit letters,
digits, periods, underscores, and hyphens only. Once a version directory exists,
the command refuses to overwrite it; use a genuinely new identity.

An implicit Git-commit version is refused when tracked files are dirty because
that commit would not describe the executing code. Commit the work first, or use
an explicit clearly local version while iterating. The package manifest records
whether the source worktree was dirty.

### 21.3 Package contents

```text
artifacts/model-packages/<version>/
├── model/                       # standard MLflow Model + pinned environment
│   ├── MLmodel
│   ├── input_example.json
│   ├── model.pkl
│   ├── python_env.yaml
│   └── requirements.txt
├── model.joblib                 # compatibility with current prediction CLI
├── metrics.json                 # selection, test gate, run, code, and data IDs
├── model-card.md
├── inference-request.json       # validated raw request fixture
├── data-manifest.json
├── data-profile.json
├── resolved-config.json
└── package-manifest.json        # identity and SHA-256/size inventory
```

MLflow may add other standard environment files under `model/`; the package
manifest inventories them too. Model-card inputs are preserved through metrics,
data evidence, and resolved configuration rather than requiring hidden MLflow
state.

### 21.4 Validate independently

```powershell
ml-template-validate-package `
  --package artifacts/model-packages/local-check
```

Validation fails when:

- a required file is absent or an unexpected file appears;
- any byte size or SHA-256 differs from the manifest;
- metrics do not show a passed quality gate;
- metrics and package metadata disagree on data identity;
- the MLflow Model lacks its signature or input example;
- the committed inference request fails through either the joblib or MLflow
  model format.

Validation checks hashes before loading serialized models. Nevertheless,
pickle/joblib/cloudpickle artifacts can execute code while loading; consume only
packages produced by a trusted workflow and stored in an access-controlled
artifact system.

### 21.5 Why an equivalent task runner instead of `MLproject`

The package command is the single task runner used locally and in CI.
`pyproject.toml` already defines the package and dependency groups, while the
exported MLflow Model captures its serving environment. Adding an `MLproject`
plus a second Conda definition now would duplicate dependency authority without
changing the execution contract.

A real organization may wrap this command with a platform-native orchestrator,
but the wrapper should invoke this entry point rather than reimplement training.

### 21.6 Training/package CI

`.github/workflows/train-package.yml` runs on manual dispatch, relevant pull
requests, and relevant pushes to `main`. It:

1. installs the same package experiment group;
2. generates and prepares the deterministic fixture;
3. calls `python -m ml_project_template.model_package` with the full commit SHA;
4. calls the independent validator;
5. uploads `model-package-<commit SHA>` for 30 days.

Concurrency cancellation prevents older builds for the same ref from wasting
resources. Workflow permissions are read-only. Pull requests may build and
upload temporary workflow artifacts for verification, but the workflow contains
no Docker/registry publishing and has no package-write permission.

This artifact retention is enough to prove the template. A real production
system normally promotes the validated directory to long-lived object storage
or a model registry under an explicit environment approval policy.

### 21.7 Download and consume a CI package

In GitHub, open the successful **Train and package model** workflow run and
download the artifact named `model-package-<full commit SHA>`. After extraction,
run the validator before serving or promotion.

The `model/` directory is the Phase 6 serving input. Deployment must download a
specific versioned bundle; it must not retrain from mutable data as part of
service startup.

### 21.8 Verify Phase 5

Run the full local quality gate:

```powershell
PowerShell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

Then perform the build and validation commands above. Confirm that:

- the package directory name equals the requested version or commit SHA;
- `package-manifest.json` lists every non-manifest file;
- `model/MLmodel` contains a signature and input-example reference;
- `metrics.json` reports `quality_gate_passed: true` and the Phase 4 data
  reference;
- rebuilding the same version fails rather than overwriting it;
- modifying a copied file makes validation fail.

---

## 22. Related documentation

- `README.md` — orientation and quick start
- `docs/current-state-assessment.md` — what is good, resolved, partial, or missing
- `docs/project-extension-plan.md` — phased architecture and implementation plan
- `docs/project-handbook.md` — detailed operating and customization reference
