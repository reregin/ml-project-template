# ML Project Template Handbook

## About this handbook

This is the detailed how-to reference for working with `ml-project-template`.
Unlike the README, which provides orientation and a quick start, this handbook
explains how the current system works, how to operate it, what to replace for a
real project, and how to diagnose common failures.

The handbook will grow with the implementation phases. It currently documents
the completed **Phase 1 core**:

- installable Python package;
- deterministic, domain-neutral example data;
- minimum data validation;
- leakage-safe train/test separation;
- one fitted preprocessing-and-estimator pipeline;
- one authoritative model artifact;
- command-line raw inference;
- optional Streamlit and notebook interfaces.

Later editions will add automated testing, CI, experiment management, data
versioning, reproducible packaging, serving, monitoring, and release workflows.

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
| `data.py` | CSV loading and minimum training-data validation |
| `example_data.py` | Reproducible synthetic dataset generator |
| `pipeline.py` | Preprocessing plus baseline estimator construction |
| `train.py` | Authoritative training and artifact-writing command |
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

Development tools reserved for the testing phase:

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

## 5. Understand the Phase 1 data contract

The minimum contract is defined by `Settings` and enforced by `data.py`.

Training currently requires:

- a non-empty CSV;
- all configured numeric and categorical columns;
- the configured target column;
- no missing target values;
- both binary target values `0` and `1`.

Missing feature values are permitted because the pipeline imputes them.

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

Generate data first, then run:

```powershell
python -m ml_project_template.train
```

Training performs these steps:

1. resolves and reads the configured CSV;
2. validates the minimum training contract;
3. selects configured features and target;
4. creates a stratified train/test split;
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
```

Or pass the intended CSV explicitly:

```powershell
python -m ml_project_template.train --data path/to/data.csv
```

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

Phase 1 intentionally does not provide:

- comprehensive automated tests;
- test coverage reporting;
- lint/type/security automation;
- pull-request CI;
- full schema and data-quality validation;
- model comparison or tuning;
- authoritative MLflow tracking and lineage;
- model-quality gates;
- data versioning;
- packaged serving artifacts;
- prediction API or container;
- monitoring, dashboards, alerts, or drift detection.

Phase 2 should add automated tests and CI around the exact manual contract
documented here. When that phase is completed, this handbook should gain:

- test command reference;
- test-fixture guidance;
- CI workflow explanation;
- failure interpretation;
- coverage and quality-tool commands;
- instructions for adding a tested feature or model implementation.

---

## 18. Related documentation

- `README.md` — orientation and quick start
- `docs/current-state-assessment.md` — what is good, resolved, partial, or missing
- `docs/project-extension-plan.md` — phased architecture and implementation plan
- `docs/project-handbook.md` — detailed operating and customization reference

