# ML Project Template Extension Plan

## 1. Purpose of this document

This document is the self-contained roadmap for evolving this repository from a
data-science starter into a reusable, progressively adoptable ML project
template. It preserves the reusable lessons from the former `ml-digdaya`
Dicoding submission so future work can continue with only this repository open.

The template must support three stopping points without forcing every project to
use every tool:

1. **DS track** — explore data, prevent leakage, preprocess, train, and evaluate.
2. **ML engineering track** — turn experiments into tested, reproducible training
   and inference code with versioned artifacts.
3. **MLOps track** — retrain in CI, package and serve the model, monitor the
   service, and alert on operational failures.

The MLOps track extends the earlier tracks; it does not replace them.

---

## 2. Core product decision

Maintain **one modular template with optional layers**, rather than separate DS
and MLOps templates. A project may stop at any track. Deployment-specific
variants should only become separate examples if their architectures genuinely
diverge, such as local Docker Compose versus Kubernetes or a cloud-managed
platform.

“Complete” means that the lifecycle is covered when all optional layers are
enabled. It must not mean that every dependency and service is installed for a
notebook-only project.

Guiding rules:

- Keep the core generic; put dataset-specific code in an example project.
- Prefer a single fitted sklearn `Pipeline` containing preprocessing and the
  estimator. This prevents training-serving skew.
- Split data before fitting imputers, encoders, scalers, feature selection, or
  any transformation that learns from data.
- Every automated stage must validate its inputs and outputs and fail loudly.
- Generated data, models, MLflow runs, screenshots, databases, environments,
  and secrets are not source code and should not be committed.
- Local execution and CI must call the same entry points.
- Operations are optional, but when enabled they must be reproducible from
  committed configuration.

---

## 3. What the current repository already has

Current structure:

```text
ml-project-template/
├── app/main.py
├── data/{raw,processed,external}/
├── models/
├── notebooks/
│   ├── 01_eda_and_discovery.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_inference_test.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── train.py
│   └── inference.py
├── .gitignore
├── README.md
└── requirements.txt
```

Useful existing ideas:

- notebooks are separated from reusable source code;
- raw, processed, and external data have distinct locations;
- the intended workflow is split-first to reduce leakage;
- preprocessing can be fitted on train and applied to test;
- MLflow integration, model metrics, plots, and a UI are started;
- configuration centralizes paths and basic parameters.

### Current problems to fix before extending

1. `src/inference.py` calls `preprocessing.clean_column_names()`, but that
   function does not exist.
2. Training saves to `config.MODEL_PATH` (`models/model.joblib`), while inference
   looks for `models/{MODEL_NAME}_v1.pkl`.
3. Training consumes already transformed CSV files and saves only the estimator.
   Inference accepts raw fields, so the fitted preprocessor is not reliably
   applied with the model.
4. `app/main.py` is a hard-coded CreditGuard example, although the repository is
   presented as a generic template.
5. Several modules modify `sys.path` at runtime. The repository should instead be
   an installable package and use normal imports.
6. `requirements.txt` is a full machine environment dump, includes many unrelated
   packages, and contains a non-portable local `file:///C:/...` dependency.
7. There are no automated tests, lint/type checks, data-contract tests, or CI.
8. The `.gitignore` does not yet cover all generated models, MLflow stores,
   processed data, logs, coverage files, or tool caches.
9. Generic classification metrics use weighted averaging, which can hide poor
   minority-class performance. Metric policy must be explicit per task.
10. Broad exception handling can convert programming errors into vague runtime
    responses. Expected validation failures and unexpected failures need distinct
    treatment.
11. The README claims “production-ready” before packaging, testing, serving,
    monitoring, security, and release behavior are implemented.

---

## 4. Reusable knowledge preserved from `ml-digdaya`

### 4.1 Experiment-to-automation flow

The submission used this progression:

1. manually load and understand an Indonesian ecommerce dataset;
2. perform EDA in a notebook;
3. manually develop preprocessing;
4. convert the stable preprocessing steps into a CLI Python script;
5. run that script in GitHub Actions;
6. upload the processed dataset as a workflow artifact.

Notebook sections were: dataset introduction, imports, data loading, EDA, and
data preprocessing.

The automated script accepted `--input` and `--output` paths and performed:

- flexible CSV loading;
- whitespace cleanup in column names;
- duplicate removal;
- normalized column-name lookup;
- target derivation from order status (`cancelled` versus `completed`);
- removal of identifiers, status/leakage fields, cancellation reason, and
  returned-quantity fields;
- datetime expansion into year, month, day, weekday, and hour;
- numeric-string cleanup and conversion;
- median numeric imputation;
- IQR-based numeric clipping;
- categorical whitespace cleanup and missing-category filling;
- one-hot encoding;
- output assertions: no missing values and exactly two target classes;
- creation of the destination directory and saving the processed CSV.

**Lesson to retain, but improve:** automation should expose a CLI and assertions,
but learned transformations such as medians, clipping bounds, and category
encodings must be fitted only on training data. The generic template should put
them inside a persisted sklearn pipeline instead of preprocessing the complete
dataset before splitting.

The preprocessing workflow was manually triggerable and ran on relevant pushes
and pull requests. It checked out code, installed Python dependencies, ran the
CLI, then uploaded a commit-SHA-named processed-data artifact with a retention
period. Preserve the path-filtering, explicit output validation, and artifact
upload patterns. Do not automatically commit generated data back to the repo.

### 4.2 Experiment tracking and tuning

The baseline training example used:

- a stratified train/test split;
- `Pipeline(StandardScaler, LogisticRegression)`;
- balanced class weights;
- MLflow sklearn autologging with signatures and input examples;
- accuracy, precision, recall, F1, and ROC AUC;
- dataset, model-type, and logging-strategy tags.

The tuned example used `RandomizedSearchCV` over logistic-regression `C`, penalty,
and class weights, with F1 scoring and cross-validation. It disabled autologging
and manually recorded:

- best parameters and search settings;
- train accuracy/F1;
- test accuracy, precision, recall, F1, ROC AUC, and log loss;
- best cross-validation F1;
- model signature and input example;
- classification report;
- dataset profile and target distribution;
- all CV results;
- confusion matrix and ROC curve;
- feature coefficients in CSV and chart form.

Reusable lessons:

- always log enough metadata to reproduce and compare a run;
- log an input example and signature with every deployable model;
- separate baseline and tuning concerns without duplicating the whole pipeline;
- choose the tuning objective and reported metrics from the business problem;
- record both aggregate metrics and diagnostic artifacts;
- compare train and validation/test performance to expose overfitting;
- never select a final model using the untouched test set.

### 4.3 Reproducible training as an MLflow Project

The submission packaged training with an `MLproject` file and a Conda environment.
Its entry point parameterized:

- input data path;
- output model path;
- test size;
- random seed.

Training produced three explicit deliverables:

```text
outputs/
├── model/                    # MLflow Model, including MLmodel metadata
├── metrics.json
└── inference_request.json   # valid dataframe_split request example
```

The saved MLflow model included a complete sklearn pipeline, signature, input
example, environment metadata, and serialised model. Training checked that the
`MLmodel` file existed before reporting success.

Example local command retained from the submission:

```powershell
mlflow run ./MLProject --env-manager local
```

For this template, the exact directory may change, but the principle remains:
one parameterized command must reproduce training and produce a self-contained,
servable model plus machine-readable evaluation and request examples.

### 4.4 Retraining CI and container publishing

The retraining GitHub Actions workflow:

- ran manually and on changes to training-project files;
- used Python 3.12 and cached pip dependencies;
- ran the same MLflow Project used locally;
- asserted the existence of `MLmodel`, `metrics.json`, and
  `inference_request.json`;
- uploaded all outputs as a SHA-specific artifact;
- skipped Docker publishing on pull requests;
- skipped publishing safely when Docker Hub secrets were absent;
- built a serving image with `mlflow models build-docker`;
- published both `latest` and immutable Git-SHA image tags;
- used `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` secrets.

Reusable lessons and improvements:

- tests and output validation must run before any image is published;
- immutable SHA tags are the deployable identity; `latest` is only a convenience;
- credentials belong in repository secrets, never committed files;
- add concurrency control and least-privilege permissions;
- pin third-party GitHub Actions to trusted versions or commit SHAs;
- add metric-quality gates before packaging;
- generate an image SBOM and scan dependencies/images when the core is stable;
- avoid publishing images for untrusted pull requests;
- make the registry configurable instead of hard-coding Docker Hub.

### 4.5 Serving and inference contract

The MLflow server accepted requests using the `dataframe_split` JSON format:

```json
{
  "dataframe_split": {
    "columns": ["feature_a", "feature_b"],
    "data": [[1.0, "category"]]
  }
}
```

The submission included a CLI load generator with:

- configurable URL, dataset, timeout, count, and interval;
- normal mode for a valid request;
- burst mode for multiple rapid requests;
- invalid mode for exercising error handling and alerts;
- response status/body reporting and a final success count.

Retain a generated valid request example and smoke-test script. For the generic
template, also add a typed API schema, readiness/liveness endpoints, request-size
limits, structured logs, and an explicit API/model version response.

### 4.6 Monitoring stack

The submission used Docker Compose with four services:

1. an MLflow model-serving image;
2. a Flask monitoring proxy/exporter;
3. Prometheus;
4. Grafana.

The proxy validated requests, forwarded them to the model server, exposed
`/predict`, `/health`, and `/metrics`, periodically pinged the model, and recorded
these metrics:

| Metric | Type | Meaning |
|---|---|---|
| `ml_inference_requests_total` | Counter | received prediction requests |
| `ml_inference_success_total` | Counter | successful prediction requests |
| `ml_inference_errors_total` | Counter | invalid or failed requests |
| `ml_inference_latency_seconds` | Histogram | end-to-end inference latency |
| `ml_inference_rows_total` | Counter | predicted input rows |
| `ml_prediction_cancelled_total` | Counter | class-1 predictions |
| `ml_prediction_not_cancelled_total` | Counter | class-0 predictions |
| `ml_last_prediction` | Gauge | most recent binary result |
| `ml_last_inference_timestamp_seconds` | Gauge | most recent successful prediction time |
| `ml_model_up` | Gauge | model-server reachability |
| `ml_inference_requests_in_progress` | Gauge | active requests |

The Grafana dashboard showed all eleven metrics and calculated p95 latency using:

```promql
histogram_quantile(
  0.95,
  sum(rate(ml_inference_latency_seconds_bucket[5m])) by (le)
)
```

The provisioned alerts were:

- model down: `ml_model_up < 1` for 30 seconds;
- high error rate: errors divided by requests above 10% for 20 seconds;
- high request volume: more than 20 requests in one minute for 10 seconds.

Prometheus scraped the exporter every five seconds. Grafana provisioned its
Prometheus datasource, dashboard directory, and alert rules from committed YAML
and JSON. Notification contact points were deliberately configured outside Git
because they contain secrets.

Important improvements for a reusable template:

- do not use domain-specific prediction names; prefer configurable class labels
  or safe bounded labels;
- avoid unbounded/high-cardinality metric labels such as user IDs or raw feature
  values;
- pin container image versions instead of using only `latest`;
- never ship default Grafana passwords as production defaults;
- distinguish infrastructure metrics, model-output metrics, data-quality metrics,
  and true performance/drift metrics;
- observed prediction counts do **not** measure accuracy without delayed ground
  truth;
- keep notification endpoints and credentials in secrets;
- add resource limits, persistent-volume guidance, and retention guidance.

### 4.7 Submission artifacts that must not shape the template

Do not copy these patterns from `ml-digdaya`:

- duplicated `SMSML_Regina-George` and `SMSML_Regina-George - Copy` folders;
- nested `.git` repositories;
- committed `.venv` directories;
- screenshots and rubric-evidence folders as product architecture;
- committed `mlruns`, SQLite tracking databases, generated model binaries, or
  output datasets;
- hard-coded student/user names, dataset names, Docker accounts, ports, class
  labels, experiment names, or dashboards;
- text files whose only purpose is holding external URLs;
- copying a full preprocessed dataset into multiple lifecycle stages;
- default credentials;
- a proxy that becomes the permanent API merely to expose metrics—instrument the
  chosen serving API directly when practical.

---

## 5. Proposed target structure

The final naming can be adjusted while implementing, but responsibilities should
remain clear:

```text
ml-project-template/
├── .github/workflows/
│   ├── ci.yml
│   ├── preprocess.yml              # optional data-artifact workflow
│   └── train-package.yml           # optional MLOps workflow
├── app/                             # optional example UI or API adapter
├── configs/
│   ├── base.yaml
│   ├── model.example.yaml
│   └── monitoring.example.yaml
├── data/
│   ├── raw/.gitkeep
│   ├── interim/.gitkeep
│   ├── processed/.gitkeep
│   └── external/.gitkeep
├── deployment/                      # optional MLOps track
│   ├── docker/
│   └── compose.yaml
├── docs/
│   ├── architecture.md
│   ├── data-contract.md
│   ├── model-card-template.md
│   └── operations-runbook.md
├── examples/
│   └── binary-classification/       # small, non-sensitive demonstration
├── monitoring/                      # optional MLOps track
│   ├── prometheus.yml
│   └── grafana/
│       ├── dashboards/
│       └── provisioning/
├── notebooks/
│   ├── 01_eda_and_discovery.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   └── 04_inference_test.ipynb
├── requirements/
│   ├── base.txt
│   ├── experiment.txt
│   ├── serving.txt
│   ├── monitoring.txt
│   └── dev.txt
├── src/ml_project_template/
│   ├── config.py
│   ├── contracts.py
│   ├── data.py
│   ├── features.py
│   ├── pipeline.py
│   ├── evaluate.py
│   ├── train.py
│   ├── predict.py
│   ├── tracking.py
│   └── serving/
│       ├── api.py
│       ├── schemas.py
│       └── metrics.py
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
├── MLproject                       # optional compatibility entry point
├── pyproject.toml
├── README.md
└── project-extension-plan.md
```

Use package/resource naming that is easy to replace when instantiating the
template. Empty optional directories alone are not valuable; each enabled layer
must include a working example, tests, documentation, and removal instructions.

---

## 6. Contracts that connect the lifecycle

The most important design work is not the folder tree; it is defining stable
contracts between stages.

### Data contract

Define and validate:

- required and optional columns;
- data types and coercion policy;
- target column and allowed values;
- nullability;
- uniqueness/identifier expectations;
- numeric bounds and categorical domains where known;
- duplicate policy;
- time-related split requirements;
- sensitive fields that must not enter features or logs.

Validation should run at ingestion, before training, and at serving input. Store
only a small synthetic fixture in Git.

### Training contract

A training run receives configuration plus a data reference and produces:

- a fitted end-to-end pipeline;
- metrics in machine-readable JSON;
- MLflow run metadata;
- model signature and input example;
- evaluation plots/tables;
- data/config/code version metadata;
- a model card or model-card inputs;
- a pass/fail result against explicit quality thresholds.

### Model artifact contract

The deployable artifact must contain or reference:

- preprocessing and estimator together;
- expected input schema;
- output semantics;
- dependency/environment information;
- model/version identity;
- training configuration and metrics;
- a valid inference example.

### Serving contract

Define:

- versioned prediction endpoint and JSON schema;
- batch versus single-row behavior;
- deterministic validation errors;
- readiness and liveness behavior;
- timeout and request-size behavior;
- response shape, probability semantics, and model version;
- privacy-safe structured logging.

### Observability contract

Start with RED-style service metrics: request rate, errors, and duration. Add
availability, rows/batch size, and bounded output-distribution metrics. Add data
quality and drift only when a reference window is defined. Add true performance
monitoring only when production ground truth can be joined safely.

---

## 7. Dependency strategy

Replace the current environment dump with intentional dependency groups. The
exact versions must be selected and locked during implementation.

- **base:** pandas, numpy, scikit-learn, joblib, configuration and validation
  library if adopted;
- **experiment:** Jupyter, matplotlib/seaborn or Plotly, MLflow;
- **serving:** FastAPI, Uvicorn, request/response validation;
- **monitoring:** Prometheus client;
- **dev:** pytest, coverage, formatter, linter, type checker, pre-commit;
- **optional models:** XGBoost, LightGBM, CatBoost, SHAP—install only for projects
  that select them.

Prefer `pyproject.toml` optional dependency groups and a reproducible lockfile.
Never retain a local filesystem URL in committed dependency metadata.

---

## 8. Phased implementation roadmap

Do the phases in order. Do not start Grafana while the saved training artifact
cannot reliably predict raw inputs.

### Phase 0 — Baseline and scope

Tasks:

- record the supported Python version and initial supported task type;
- begin with tabular binary classification, then add regression/multiclass only
  through task-specific adapters and tests;
- choose the small synthetic/example dataset;
- document commands currently expected to work;
- convert this plan into tracked issues or a checklist if desired.

Acceptance criteria:

- template scope and non-goals are explicit;
- no real/private dataset is required for the smoke test;
- one command identifies the current failures before refactoring.

Learning focus: distinguish a reusable template from a single finished model.

### Phase 1 — Repair and package the core

**Status: completed and manually verified on 2026-08-06.** The implementation
uses the installable `src/ml_project_template/` package, a deterministic neutral
binary-classification fixture, and one fitted sklearn pipeline artifact. Phase 2
will convert the manual smoke checks into automated tests and CI.

Tasks:

- add `pyproject.toml` and move code under `src/ml_project_template/`;
- remove all `sys.path` mutation;
- replace hard-coded CreditGuard UI/configuration with a generic example;
- define one authoritative artifact path;
- build and persist one end-to-end sklearn pipeline;
- make prediction load exactly that artifact;
- fix probability handling so a legitimate probability of `0.0` is preserved;
- use `pathlib.Path` for portable paths;
- introduce typed configuration and structured logging;
- update `.gitignore` for data outputs, models, MLflow, environments, logs, and
  caches;
- replace the requirements dump with dependency groups.

Acceptance criteria:

- clean-environment installation works;
- `python -m ...train` creates the documented artifact;
- `python -m ...predict` accepts raw input and returns a prediction;
- preprocessing is fitted on train only and is embedded in the artifact;
- no machine-specific path is committed.

Learning focus: imports, packaging, pipelines, reproducibility, and
training-serving skew.

### Phase 2 — Tests and quality automation

Tasks:

- unit-test data loading, contract validation, feature construction, metrics, and
  prediction;
- integration-test tiny end-to-end training and inference;
- assert output schema and feature-order behavior;
- test missing columns, unknown categories, invalid target, empty data, and model
  not found;
- add lint, format, type, security, and test commands;
- add a fast CI workflow for pushes and pull requests;
- generate coverage without treating a percentage alone as quality.

Acceptance criteria:

- CI runs from a clean checkout without production data;
- a training/inference smoke test proves artifact compatibility;
- deliberate schema and model-path regressions fail tests;
- CI has least-privilege permissions and no required secrets.

Learning focus: tests as executable lifecycle contracts.

### Phase 3 — Experiment management and evaluation

Tasks:

- centralize experiment configuration;
- record parameters, task-appropriate metrics, tags, signatures, examples, and
  diagnostic artifacts in MLflow;
- provide baseline and tuning entry points sharing the same pipeline builder;
- use train/validation/test or cross-validation correctly;
- add configurable quality gates;
- create a model-card template;
- record code revision, data reference/hash, and configuration snapshot;
- document local MLflow UI usage and remote-tracking configuration.

Acceptance criteria:

- two runs can be compared and reproduced from recorded inputs;
- the chosen model is selected without test-set tuning;
- every deployable model has a signature and valid request example;
- failed quality gates prevent packaging.

Learning focus: experiments are evidence, not just saved model files.

### Phase 4 — Data automation and versioning

Tasks:

- expose ingestion/preprocessing as idempotent CLIs;
- validate raw input and processed output;
- produce a data profile and provenance metadata;
- add path-filtered preprocessing CI if useful;
- upload generated data as an artifact rather than committing it;
- decide whether DVC/object storage is justified; do not add DVC solely because
  it appeared in the old environment;
- support time-based splitting when the problem requires it.

Acceptance criteria:

- the same command produces the same output from the same versioned input/config;
- learned preprocessing never sees validation/test data;
- data lineage connects a model run to an immutable input reference;
- invalid data stops before training.

Learning focus: repeatability, lineage, and leakage boundaries.

### Phase 5 — Reproducible training and packaging

Tasks:

- add a parameterized MLflow Project or equivalent task runner;
- output the MLflow Model, `metrics.json`, request example, and model-card inputs;
- validate all required outputs;
- add retraining CI with artifact retention;
- tag artifacts with the commit SHA;
- add concurrency controls to avoid redundant runs;
- make registry/image publication conditional and environment-configurable.

Acceptance criteria:

- local and CI training call the same entry point;
- output validation fails a broken run;
- a SHA-addressable artifact can be downloaded and served;
- pull requests never publish release images.

Learning focus: CI proves reproducibility; it is not merely a remote shell.

### Phase 6 — Serving and containerization

Tasks:

- implement a small typed FastAPI serving layer or document direct MLflow serving;
- add `/live`, `/ready`, `/predict`, and `/metrics` as appropriate;
- return model/version metadata;
- provide a minimal, pinned, non-root container;
- add container health checks;
- add valid, burst, and invalid smoke/load requests;
- provide local Compose wiring without embedding credentials;
- document timeouts, scaling limits, and failure modes.

Acceptance criteria:

- a fresh build can serve the CI-produced artifact;
- valid input succeeds and invalid input returns a stable 4xx response;
- readiness fails when the model cannot be loaded;
- container and local Python predictions match on a fixture;
- no secret or default production password is committed.

Learning focus: a model artifact becomes a versioned network service.

### Phase 7 — Monitoring, dashboards, and alerts

Tasks:

- instrument rate, errors, duration, availability, rows, and in-progress requests;
- keep labels bounded and privacy-safe;
- provision Prometheus and a generic Grafana dashboard;
- adapt the three preserved alert patterns with configurable thresholds;
- add an operations runbook for model down, error spike, and latency spike;
- document how to configure notification contact points outside Git;
- separately design data-quality, output drift, feature drift, and delayed-ground-
  truth performance monitoring.

Acceptance criteria:

- normal, burst, invalid, and model-down scenarios visibly change metrics;
- dashboards load from provisioning files;
- each alert can be tested deliberately and has a runbook action;
- prediction counters are not mislabeled as model accuracy;
- monitoring survives service restarts according to documented retention.

Learning focus: distinguish system health from statistical model health.

### Phase 8 — Release and template ergonomics

Tasks:

- document DS-only, ML-engineering, and full-MLOps quick starts;
- document how to remove/disable optional layers;
- add a template customization checklist for project name, package name, target,
  task type, schema, metrics, thresholds, ports, registry, and owners;
- add changelog/versioning and release automation;
- consider a copier/cookiecutter-style generator only after the manual structure
  is stable;
- add dependency/image scanning and an SBOM;
- test the template by instantiating a brand-new example project.

Acceptance criteria:

- a new user can run the DS track without Docker/Grafana dependencies;
- a new user can enable the full stack from documented commands;
- placeholder names are detected or clearly enumerated;
- generated project CI passes without access to the template repository.

Learning focus: templates are products and require usability testing.

---

## 9. Recommended CI workflow separation

Keep feedback fast by separating concerns:

1. **`ci.yml`** — lint, types, unit tests, and tiny integration test on every PR.
2. **`preprocess.yml`** — optional, path-filtered data validation/transformation;
   uploads data artifacts.
3. **`train-package.yml`** — manual, scheduled, or relevant-path retraining;
   applies quality gates, uploads model artifacts, and optionally publishes an
   image outside pull requests.
4. **release/deploy workflow** — add later; promote a specific immutable artifact
   instead of retraining during deployment.

This separation prevents expensive retraining from slowing every code change and
keeps model creation distinct from environment promotion.

---

## 10. Definition of done for the complete template

The repository is “complete” only when:

- DS-only use is documented and does not require operational dependencies;
- data splitting and learned transformations are leakage-safe;
- an end-to-end pipeline is reproducibly trained, evaluated, and saved;
- raw inference uses exactly the fitted training transformations;
- tests enforce data, training, artifact, and API contracts;
- MLflow records reproducibility metadata, signatures, examples, and artifacts;
- CI validates code and can produce an immutable model artifact;
- a container can serve that exact artifact;
- metrics, dashboards, and alerts are reproducibly provisioned;
- operational and statistical monitoring are explained separately;
- generated files and secrets stay out of Git;
- all hard-coded example identities can be replaced systematically;
- a clean template instantiation passes its own quick start and CI.

---

## 11. Decisions to make during implementation

Do not block Phase 1 on all of these, but record each decision when reached:

- configuration library and file format;
- packaging/lock tool;
- first-class serving option: FastAPI, MLflow serving, or both;
- MLflow local store versus optional remote store instructions;
- artifact and image registry abstraction;
- whether MLflow Projects remain the runner or a simpler task command becomes
  primary;
- supported problem types beyond initial tabular binary classification;
- schema validation library;
- whether data versioning is needed in the core or only documented as an option;
- model promotion states and approval rules;
- ground-truth collection/join strategy for performance monitoring;
- local Compose only versus later cloud deployment examples.

---

## 12. Suggested next working session

Start with **Phase 0 and Phase 1 only**:

1. run the current tiny example and capture its failures;
2. choose the generic example dataset and input schema;
3. introduce `pyproject.toml` and the installable package layout;
4. unify preprocessing plus estimator into one artifact;
5. make training and raw inference agree;
6. replace the dependency dump;
7. add the first end-to-end smoke test before extending infrastructure.

Do not begin by copying the old Docker Compose stack. A reliable artifact and
inference contract are prerequisites for every later layer.
