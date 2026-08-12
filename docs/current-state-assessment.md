# Current-State Assessment

## 1. Why this document exists

This document explains `ml-project-template` as it exists today and preserves
the important findings from its original pre-extension state. It answers four
questions:

1. What does the current repository already provide?
2. Which parts are genuine best practices?
3. Which parts are only promising scaffolding, incomplete, or currently broken?
4. Which extension-plan phase will address each gap?

This is an assessment, not a claim that the current repository is already a
production-ready system.

---

## 2. Honest summary

The current repository is a **useful tabular data-science starter with a tested
complete Phase 8 progressive template core**. Its strongest ideas are separation of
experimentation from reusable code, a split-first leakage policy, structured
data directories, an installable package, and one raw-input-to-prediction
pipeline artifact.

It is not yet a complete reusable ML template. Phase 1 repaired package,
dependency, training, artifact, and inference consistency. Phase 2 now enforces
those contracts with unit/integration tests, coverage, formatting, linting, type
checking, security scanning, and CI. Phase 3 now adds MLflow comparison,
development-only selection, final-test isolation, run lineage, signatures,
diagnostics, model cards, and a packaging quality gate. Phase 4 now adds a
versioned data contract, deterministic preparation, profiles, content-addressed
lineage, manifest verification, temporal split support, and path-filtered data
CI. Phase 5 now adds a shared local/CI package command, standard MLflow Model
bundle, commit-addressed identity, complete output validation, atomic
publication, and retained CI artifacts. Phase 6 adds a typed prediction API,
separate liveness/readiness behavior, stable errors, package metadata, hardened
container/Compose wiring, and CI smoke/parity checks. Phase 7 adds privacy-safe
service metrics, provisioned Prometheus/Grafana, persistent local history,
tested alert rules, response runbooks, and an explicit boundary between
operational and statistical model health. Deployment and registry promotion
remain project-specific work. Phase 8 adds progressive quick starts, safe
standalone instantiation, customization/removal guidance, template versioning,
release assets, dependency/image security checks, and SBOM generation.

Current maturity by track:

| Track | Current status | Explanation |
|---|---|---|
| DS experimentation | Good foundation | Ordered notebooks, data layout, EDA prompts, split-first intent, preprocessing and training scaffolding exist. |
| ML engineering | Complete reusable baseline | The lifecycle is protected by 97 tests and at least 90% branch coverage; fresh renamed projects execute without the template checkout. |
| MLOps | Complete local reference track | Packaging, serving, Prometheus/Grafana, alert tests, runbooks, scans, and SBOMs exist; registry promotion and production deployment remain project-specific. |

The right strategy is to preserve this now-working baseline and add tests,
experimentation, automation, and optional MLOps layers in order.

---

## 3. Current repository map

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
│   └── project-extension-plan.md
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

The intended current flow is:

```text
domain-neutral raw example data
   ↓
versioned contract validation + deterministic canonicalization
   ↓
prepared CSV + profile + SHA-256 lineage manifest
   ↓
manifest verification → stratified random or chronological test split
   ↓
preprocessing fitted on the training split only
   ↓
one sklearn Pipeline containing preprocessing + estimator
   ↓
quality-gated MLflow Model + metrics/card/request/config/data evidence
   ↓
immutable artifacts/model-packages/<commit SHA>/ bundle
   ↓
raw JSON inference → optional Streamlit application
```

Phase 1 verifies this connection end-to-end. Prediction loads the exact artifact
written by training and accepts the same raw feature schema, including unseen
categorical values through the fitted encoder.

---

## 4. What is already good practice

### 4.1 Separating notebooks from reusable source code

**Status: good practice.**

Keeping exploratory notebooks in `notebooks/` and reusable logic in `src/` is a
strong design decision. Notebooks are appropriate for discovery, visualization,
and explaining decisions; repeatable production logic belongs in functions and
command-line entry points.

Why it matters:

- experimentation remains readable;
- stable logic can be tested;
- CI can call Python modules without executing an entire notebook;
- the same logic can eventually be shared by training and inference.

The notebook is still a laboratory, not merely a viewer for `src/`. It is allowed
to contain an alternative implementation that may never become shared code. The
real concern is **inconsistency and unclear authority**, not duplication itself.

The working rule is:

> Notebooks may contain experimental implementations, but established project
> behavior should have one authoritative implementation in shared code.

#### What notebooks may contain as experiments

Notebook-local work can include:

- new feature-engineering hypotheses;
- alternative imputers, encoders, scalers, selectors, or samplers;
- different model families and hyperparameter-search ideas;
- experimental split or validation strategies;
- alternative metrics and decision thresholds;
- temporary debugging and data-quality investigations;
- exploratory visualizations and interpretation techniques;
- quick implementations used to decide whether an idea is worth engineering;
- rejected experiments retained with notes explaining what was learned.

This code does not need to move into `src/` merely because it exists. Most
experiments may be discarded. It should, however, be labelled as experimental
and compared under a consistent data and evaluation setup.

#### What shared code should do or be

Shared code is the authoritative, reusable implementation of behavior that the
project currently trusts. It should provide:

- stable data loading and schema validation;
- the official split strategy and random seeds;
- reusable feature and preprocessing components;
- the current baseline or accepted training pipeline;
- authoritative metric definitions and evaluation helpers;
- experiment-tracking configuration;
- artifact saving/loading and inference contracts;
- configuration, logging, and error behavior;
- code used by automated tests, CI, training jobs, and serving.

Shared code should be importable, parameterized, tested, and usable without
copying it into a notebook. A notebook can execute it, compose its components in
a new way, clone a pipeline and change parameters, inspect intermediate results,
or compare it with a notebook-local candidate. Importing shared code therefore
does not reduce a notebook to merely inspecting source code.

#### What initial shared code should contain before a good hypothesis exists

At the beginning of a project, there may be no proven feature engineering or
strong model choice. Shared code should then be deliberately small and boring.
It generally contains:

- a reliable loader;
- a minimal data contract and target check;
- the correct leakage-safe split for the problem;
- a simple baseline preprocessor;
- a simple baseline estimator appropriate to the task;
- a consistent evaluation function;
- configuration and deterministic seeds;
- basic MLflow/run logging;
- a standard artifact and prediction interface.

This initial shared code is not presented as the best model. It is the stable
control against which notebook hypotheses are tested. For example, a notebook
can import the baseline pipeline, create an experimental feature transformer and
XGBoost candidate locally, and evaluate both through the same shared evaluation
function.

#### When a notebook implementation becomes a problem

Duplication becomes dangerous when, for example:

- `src/preprocessing.py` contains the supposedly official preprocessing;
- a training notebook silently uses a slightly different implementation;
- nobody identifies which version is authoritative;
- the notebook's results are reported or its model is selected;
- inference still uses the `src/` implementation.

Both versions may run, yet the reported experiment no longer describes the
system that will actually make predictions. That training-serving inconsistency
is the failure to prevent.

#### Experiment promotion path

Use this lifecycle:

```text
shared baseline
      ↓
notebook-local hypothesis
      ↓
comparison using consistent validation and metrics
      ↓
rejected ───────────────→ retain notes/results if useful
      │
      └── accepted ─────→ refactor into shared code
                               ↓
                        add automated tests
                               ↓
                    training and inference use it
                               ↓
               notebook imports the promoted version
```

In short: **keep stable mechanics shared, keep uncertain hypotheses local to
notebooks, and promote successful hypotheses into tested shared code**. Phases
1–3 of the extension plan establish this boundary and promotion workflow.

### 4.2 Ordered lifecycle notebooks

**Status: good learning structure; execution still needs validation.**

The four notebooks express a useful progression:

1. EDA and discovery;
2. preprocessing;
3. model training;
4. inference testing.

The EDA notebook prompts analysis of the target, distributions, missingness,
relationships, outliers, time, cardinality, class imbalance, and
multicollinearity. Those are useful prompts for tabular modelling.

All notebook code cells are currently unexecuted in the committed files. That is
not inherently wrong—clean notebooks reduce noisy diffs—but it means the
repository itself does not prove that they run end-to-end. CI-friendly notebook
smoke testing or equivalent module tests will be introduced in Phase 2.

### 4.3 Clear data zones

**Status: good practice.**

Separating `data/raw`, `data/processed`, and `data/external` communicates data
lineage better than placing every CSV in the project root.

- `raw` should contain immutable source snapshots or local references;
- `processed` should contain derived model-ready outputs;
- `external` should contain third-party reference data.

What remains:

- add an `interim` layer if multi-step transformation needs it;
- define what is committed versus generated;
- add data contracts, provenance, version references, and validation;
- ensure ignored directories retain `.gitkeep` placeholders if desired.

These are now addressed for the generic binary-classification scope in Phase 4.
Dataset-specific semantic rules and durable remote retention remain project
responsibilities.

### 4.4 Split-first leakage prevention

**Status: correct and important best-practice principle.**

The README and code intend to split the raw dataset before fitting learned
preprocessing. `preprocessing.py` fits its transformer on `X_train` and only
transforms `X_test`, which is correct.

This prevents test-set information from entering:

- imputation statistics;
- scaling parameters;
- encoded category vocabularies;
- model fitting.

Important qualification: “split first” is not universally equivalent to a
random 80/20 split. Time-dependent, grouped, repeated-entity, or spatial data may
need temporal or group-aware splitting. The template should make split strategy
configurable rather than teaching random splitting as universally correct.

This principle will be preserved and generalized in Phases 1 and 4.

### 4.5 Using sklearn preprocessing components

**Status: good technical direction.**

The current preprocessing module uses:

- `Pipeline` for ordered transformations;
- `ColumnTransformer` for numeric and categorical features;
- `SimpleImputer` for missing values;
- `StandardScaler` for numeric values;
- `OneHotEncoder(handle_unknown="ignore")` for categorical values.

These are appropriate tools because their learned state can be fitted on
training data and reused later. Handling unknown categories is particularly
important for inference data.

What is not yet best practice is saving the preprocessor and estimator as
loosely coordinated artifacts while inference does not consistently load both.
Phase 1 will combine them into one end-to-end fitted model pipeline.

### 4.6 Central configuration

**Status: good idea, early implementation.**

`src/config.py` centralizes paths, target name, random state, split ratio,
preprocessing options, task type, and model name. Centralization prevents values
from being duplicated across every script.

What remains:

- replace “CHANGE THIS” source edits with a validated user configuration;
- use portable `pathlib.Path` values;
- separate stable defaults from environment-specific values;
- validate combinations such as task type and metric selection;
- store the resolved configuration with every run.

This is addressed in Phases 1 and 3.

### 4.7 Multiple input formats in the loader

**Status: useful feature, incomplete validation.**

The loader supports CSV, Excel, and Parquet and produces clear file-not-found and
unsupported-format failures. This is a practical starting point.

What remains is validating the *content*, not just reading the file: required
columns, data types, nullability, duplicates, target values, sensitive fields,
and empty datasets. This is addressed in Phases 1, 2, and 4.

### 4.8 Stratified splitting for likely classification targets

**Status: useful safeguard, but heuristic.**

The loader tries to preserve target proportions for categorical or low-cardinality
targets. This is helpful for imbalanced classification.

However, inferring task type from dtype or fewer than twenty unique values can
misclassify ordinal/regression problems. The already configured `TASK_TYPE`
should drive this behavior explicitly. Phase 1 will make the task contract
authoritative.

### 4.9 Initial model evaluation coverage

**Status: good baseline coverage.**

`train.py` includes classification metrics, regression metrics, confusion-matrix
generation, feature importance, and train/test evaluation. It supports several
common model families and logs model artifacts and metrics to MLflow when
available.

This is a stronger starting point than training a model and reporting only
accuracy.

What remains:

- define metrics per problem rather than using one generic policy;
- avoid weighted averages hiding minority-class failure;
- introduce validation/CV for model selection and reserve test for final use;
- add probability calibration and threshold selection when relevant;
- add quality gates and model-card evidence.

These core comparison and gate concerns are addressed in Phase 3. Calibration,
threshold policy, and task-specific business metrics remain project decisions.

### 4.10 MLflow tracking

**Status: Phase 3 best-practice local baseline.**

The training code records parameters, metrics, figures, feature importance, and
a model in MLflow. Experiment tracking is a best practice when it records enough
context to reproduce and compare runs.

The explicit experiment command now logs comparable baseline and tuned runs,
development CV metrics, search results, a data hash, Git revision, resolved
configuration, dataset profile, selection evidence, final diagnostics, and the
complete selected pipeline. The deployable MLflow model has a signature and
input example. SQLite provides local run tracking, not model promotion.

Phase 5 now provides reproducible training/package execution, immutable artifact
identity, and retained CI bundles. A long-lived registry and explicit
promotion/release policy remain later deployment decisions.

### 4.11 A separate application boundary

**Status: good architectural boundary with a generic Phase 1 example.**

Putting UI code in `app/` rather than mixing it into training logic is correct.
The application calls an inference function rather than training a model inside
the request path.

The previous CreditGuard-specific form has been replaced with neutral numeric
and categorical fields. It deliberately includes an unseen-category option to
exercise the saved encoder. The UI remains an optional client; Phase 6 now
provides the proper typed serving contract for network consumers.

### 4.12 One authoritative fitted artifact

**Status: Phase 1 best-practice baseline.**

Training now saves one sklearn `Pipeline` containing its fitted preprocessor and
estimator to `models/model.joblib`. Prediction loads that exact path, validates
required raw fields, and calls the pipeline directly. This repairs the original
training-serving mismatch.

Phase 6 now loads one validated package per process, reports its version, and
keeps failed model loading visible through readiness. Hot reload and
production-specific autoscaling remain explicit platform decisions.

---

## 5. What is promising but should not yet be called best practice

### 5.1 Explicit Phase 1 feature settings

The baseline explicitly identifies numeric and categorical columns and Phase 4
now enforces required fields, numeric coercion/finite values, missing-value
policy, categorical blanks/cardinality, duplicate rows, target support, and an
optional temporal field. Domain ranges, sensitive-field policy, and semantic
constraints still require project-specific configuration or validators.

### 5.2 One simple estimator

The baseline intentionally supports only logistic regression for tabular binary
classification. This is a stable control, not a claim that it is the best model.
Additional estimators belong in notebook experiments first and should enter
shared code only with task-specific evaluation and tests.

### 5.3 Experiment tracking is an explicit optional layer

MLflow remains in the `experiment` dependency group, so core training and raw
inference stay lightweight. Projects that install the layer call the explicit
experiment command and receive deterministic tracking, selection, evidence, and
gate behavior; nothing changes silently based on whether MLflow imports happen
to be installed.

### 5.4 Saving transformed train/test CSV files in experimental notebooks

Processed datasets can be valuable for inspection, caching, and lineage, but
they should not become the only bridge between training and inference. The
deployable artifact must retain the learned raw-to-prediction transformation.

### 5.5 A generic Streamlit demonstration

A UI makes the raw prediction path tangible, and the current form is now
domain-neutral. It remains an optional consumer—not the serving contract itself.
A real project should replace or remove it rather than treating it as an API.

---

## 6. Concrete defects found in the original implementation

These were not theoretical production enhancements; they prevented the original
project from working consistently. All twelve items below were addressed by
Phase 1 on 2026-08-06. They remain documented to explain why the refactor was
necessary.

### 6.1 Training and inference use different model paths

Training saves `config.MODEL_PATH`, currently `models/model.joblib`. Inference
instead searches for `models/{MODEL_NAME}_v1.pkl`. A successful training run can
therefore still lead to “model not found” during inference.

**Resolved:** training and prediction now share `models/model.joblib`.

### 6.2 Inference calls a nonexistent function

`inference.py` calls `pp.clean_column_names(df)`, but `preprocessing.py` defines
no `clean_column_names` function.

**Resolved:** ad hoc inference preprocessing was removed; prediction calls the
saved end-to-end pipeline.

### 6.3 Fitted preprocessing is not reliably used during inference

The current training path loads already processed CSVs and trains an estimator.
The inference path receives raw fields but does not load the saved
`preprocessor.joblib`. Even after fixing the missing function, feature count,
order, scaling, and encoding may differ.

**Resolved:** preprocessing and estimator are embedded in one artifact. Phase 2
will turn the verified smoke path into an automated integration test.

### 6.4 Probability zero is handled incorrectly

Inference returns `0.0` through a truthiness expression. Although the displayed
value happens to remain zero, truthiness is the wrong test because `0.0` is a
valid probability, not a missing value. The code should explicitly test
`probability is not None`.

**Resolved:** probability presence is checked explicitly against `None`.

### 6.5 Runtime `sys.path` modification

Several files append the project root to `sys.path`. This depends on how scripts
are launched and is a sign that the code is not installed as a package.

**Resolved:** `pyproject.toml`, the `src/ml_project_template/` package, editable
installation, and normal imports replace path mutation.

### 6.6 Non-portable dependency metadata

`requirements.txt` is an environment export rather than a minimal project
definition. It installs unrelated infrastructure and model libraries and
contains a Windows-local `file:///C:/...` package reference that another machine
cannot resolve.

**Resolved:** core, experiment, application, and development dependencies are
declared intentionally in `pyproject.toml`; operational groups will be added
when their layers exist.

### 6.7 The README is incomplete and overstates maturity

The README calls the template production-ready, although contracts, tests, CI,
packaging, serving, deployment, monitoring, secrets, and releases are absent. Its
final dependency code block also appears unfinished.

**Resolved:** the README states the supported tracks and remaining project-
specific decisions; Phase 8 adds the final template ergonomics and release layer.

### 6.8 Generic configuration conflicts with a domain-specific application

The configuration defaults to a generic `target`, while the Streamlit UI expects
credit-limit, age, billing, and payment fields and identifies itself as
CreditGuard. A new project cannot use both unchanged.

**Resolved:** the application and fixture use neutral feature/category names.

### 6.9 Feature-type detection is too narrow

The current selector explicitly recognizes `int64`, `float64`, `object`, and
`category`. It can miss nullable integers, booleans, strings, other numeric
widths, and datetime columns.

**Resolved for the baseline:** feature roles are explicit typed settings rather
than dtype guesses. Phase 2 tests them; Phase 4 expands the schema contract.

### 6.10 Dense one-hot output may exhaust memory

`OneHotEncoder(sparse_output=False)` forces a dense matrix. High-cardinality
categorical data can expand dramatically.

**Resolved for the baseline:** one-hot output remains sparse. Phase 4 adds
cardinality checks and project-specific representation guidance.

### 6.11 Exceptions are handled too broadly

Inference catches every exception and returns its raw text. This can conceal code
defects and expose internal implementation details to an application user.

**Resolved for the command-line baseline:** expected input failures are explicit
and logs use structured key-value fields. Phases 2 and 6 add deeper tests and
stable API errors.

### 6.12 Model serialization and MLflow logging are not one contract

Training writes a joblib model and separately logs an MLflow model. It is not
clearly defined which artifact is authoritative for inference or deployment.

**Resolved for Phase 1:** joblib contains the authoritative complete pipeline and
metrics are machine-readable. Phase 5 extends this into the packaged deployment
artifact contract.

---

## 7. What is missing and where it will be added

| Missing capability | Why it matters | Extension-plan location |
|---|---|---|
| **Completed:** initial binary-classification scope | “Generic” behavior otherwise becomes untestable | Phase 1 baseline |
| **Completed:** installable Python package | Removes path hacks and standardizes commands | Phase 1 |
| **Completed:** unified preprocessing + estimator artifact | Prevents training-serving skew | Phase 1 |
| **Partial:** validated typed configuration | Prevents incompatible or hidden settings | Phase 1; expanded Phase 3 |
| **Completed:** minimal/optional dependency groups | Keeps DS-only use lightweight and portable | Phase 1 |
| **Completed for the example:** versioned data/schema contract | Detects malformed or incompatible training input before fitting | Phase 4 |
| **Completed:** automated unit and integration tests | Turns assumptions into executable guarantees | Phase 2 |
| **Completed:** pull-request CI | Proves clean-checkout behavior continuously | Phase 2 |
| **Completed:** repository/notebook contract checks | Shows that the documented learning surface remains structurally valid | Phase 2 |
| **Completed:** model-selection validation strategy | Prevents tuning on the final test set | Phase 3 |
| **Completed:** MLflow signature and input example | Defines deployable input/output behavior | Phase 3 |
| **Partial:** code/data/config lineage | Records Git, data SHA, settings, and profiles; immutable data storage remains | Phases 3 and 4 |
| **Partial:** quality gates | Stops weak models from local/MLflow packaging; release gates remain | Phases 3 and 5 |
| **Completed:** model card | Records intended use, limitations, and evidence | Phase 3 |
| **Completed:** automated data pipeline | Converts stable non-learned preparation into repeatable execution | Phase 4 |
| **Partial:** data version/provenance strategy | SHA manifests and CI artifacts cover the fixture; real durable storage remains project-specific | Phase 4 |
| **Completed:** parameterized reproducible training job | Makes local and CI execution identical | Phase 5 |
| **Completed:** retraining workflow and artifact retention | Produces downloadable traceable model versions | Phase 5 |
| **Partial:** immutable artifact/image tags | Model bundles and CI smoke images use commit SHA; registry publication remains deployment-specific | Phases 5 and 6 |
| **Completed:** typed prediction API | Establishes a stable service boundary | Phase 6 |
| **Completed:** liveness/readiness and health checks | Distinguishes process health from model readiness | Phase 6 |
| **Completed:** container build and Compose example | Makes the serving environment reproducible | Phase 6 |
| **Completed:** load/smoke scenarios | Exercises normal, burst, invalid, and failure behavior | Phase 6 |
| **Completed:** service metrics | Measures request rate, errors, duration, rows, work, outputs, and availability | Phase 7 |
| **Completed:** provisioned dashboards | Makes observations reproducible | Phase 7 |
| **Completed:** tested alerts and runbooks | Turns signals into operational action | Phase 7 |
| **Completed as design boundary:** drift/data-quality strategy | Separates operational, output, feature, and delayed-ground-truth health | Phase 7 |
| **Completed:** progressive quick starts | Lets users stop at DS, ML, or MLOps layers | Phase 8 |
| **Completed:** template customization/removal guide | Prevents copied placeholders and unnecessary components | Phase 8 |
| **Completed:** release/version strategy | Treats the template itself as a maintained product | Phase 8 |
| **Completed:** security scanning and SBOM | Improves dependency and container supply-chain visibility | Phase 8 |

---

## 8. Original file-by-file interpretation and Phase 1 replacements

The detailed entries below preserve the assessment of the original files. Phase
1 replaced their authoritative responsibilities as follows:

| Current file | Authoritative responsibility |
|---|---|
| `pyproject.toml` | package metadata, core dependencies, optional groups, and CLI entry points |
| `src/ml_project_template/config.py` | typed settings, portable paths, and log configuration |
| `src/ml_project_template/data.py` | minimum training-data loading and validation contract |
| `src/ml_project_template/example_data.py` | deterministic, domain-neutral smoke-test dataset |
| `src/ml_project_template/pipeline.py` | one raw-input preprocessing and estimator pipeline |
| `src/ml_project_template/train.py` | split-first fitting, evaluation, and authoritative artifact writing |
| `src/ml_project_template/model_package.py` | immutable bundle assembly, integrity validation, and smoke inference |
| `src/ml_project_template/predict.py` | raw-record validation, artifact loading, and prediction |
| `examples/binary_classification/sample_request.json` | portable inference smoke request with an unseen category |
| `app/main.py` | optional generic Streamlit client |

The old direct modules such as `src/train.py` and `src/inference.py` no longer
exist. Their descriptions below explain what was assessed and why it changed.

### `README.md`

Role today: architectural introduction and leakage-prevention explanation.

Keep:

- separation-of-concerns explanation;
- split-first teaching;
- basic MLflow introduction.

Change:

- remove premature “production-ready” language;
- document progressive tracks and verified commands;
- repair formatting/encoding;
- distinguish experiment tracking from model registry/version promotion;
- add configuration and extension/removal instructions.

### `notebooks/01_eda_and_discovery.ipynb`

Role today: broad tabular EDA checklist.

Keep: target, distribution, missingness, outlier, cardinality, imbalance, time,
and multicollinearity prompts.

Change: drive it from the shared loader/config, ensure EDA uses only permitted
training data, and make optional analyses conditional on schema/task.

### `notebooks/02_preprocessing.ipynb`

Role today: demonstrates split, feature groups, transformer fitting, and saving.

Keep: fit-on-train/transform-test concept.

Change: make the notebook demonstrate the same pipeline builder used by training
instead of producing a separate transformation path.

### `notebooks/03_model_training.ipynb`

Role today: demonstrates processed-data loading, model parameters, evaluation,
and MLflow.

Keep: experiment explanation and comparison interface.

Change: use shared training/evaluation functions, proper validation/model
selection, signatures, lineage, and quality gates.

### `notebooks/04_inference_test.ipynb`

Role today: intended final inference validation.

Keep: a human-readable demonstration that raw input becomes a prediction.

Change: call the authoritative prediction API/artifact and pair it with an
automated integration test.

### `src/config.py`

Role today: module-level constants for paths and modelling options.

Keep: one source of configuration.

Change: validated configuration objects/files, portable paths, resolved-config
logging, and environment-specific overrides.

### `src/data_loader.py`

Role today: file loading, target cleanup, splitting, and processed-data loading.

Keep: format dispatch and clear stage boundaries.

Change: explicit task/split strategy, schema validation, no path mutation, and
separation of ingestion from dataset splitting.

### `src/preprocessing.py`

Role today: feature grouping, transformer construction, fitting, transformation,
and artifact/CSV saving.

Keep: sklearn `Pipeline`/`ColumnTransformer` and fit-on-train behavior.

Change: schema-aware features, sparse/dense policy, one deployable pipeline, and
tests for unseen/missing/mistyped inputs.

### `src/train.py`

Role today: estimator factory, evaluation, plots, MLflow logging, and model save.

Keep: explicit metrics, diagnostic plots, experiment tracking, and CLI-capable
module organization.

Change: train raw-to-prediction pipeline, validation/CV, authoritative artifact,
signatures/examples, lineage, quality gates, task adapters, and smaller focused
modules.

### `src/inference.py`

Role today: model caching and prediction response formatting.

Keep: a small public prediction function and avoiding repeated deserialization.

Change urgently: fix nonexistent preprocessing call, artifact path mismatch,
raw-input transformation, probability handling, error taxonomy, model metadata,
and readiness behavior.

### `app/main.py`

Role today: CreditGuard-specific Streamlit demonstration.

Keep: UI/application separation from training.

Change: move the concrete domain into `examples/`, generate/declare fields from
an example schema where practical, and call a stable serving/prediction contract.

### `requirements.txt`

Role today: frozen local environment.

Keep: version constraints are valuable when intentionally produced.

Replace: use declared dependency groups plus a portable lock strategy. Do not
carry forward the local filesystem dependency or unrelated packages.

### `.gitignore`

Role today: ignores common Python caches, environments, `.env`, and raw data.

Keep: protecting raw data and secrets from accidental commits.

Extend: processed/interim outputs, model binaries, MLflow runs/databases, logs,
coverage, build outputs, and tool-specific caches, while preserving required
empty-directory placeholders.

---

## 9. Best-practice distinctions to remember while learning

Some ideas are easy to confuse; these distinctions should guide future changes:

- **A clean directory tree is organization, not reproducibility.** Reproducibility
  also needs dependencies, configuration, code/data identity, and executable
  commands.
- **A saved estimator is not necessarily a deployable model.** Raw-input
  preprocessing, schema, environment, signature, and output semantics also
  belong to the artifact contract.
- **MLflow tracking is not automatically model versioning or promotion.** A
  lifecycle policy must decide which run becomes a candidate or production model.
- **Random splitting is not always leakage-safe.** The split must respect time,
  groups, and real prediction conditions.
- **Many metrics are not automatically good evaluation.** Metrics must reflect
  class imbalance, error costs, calibration, and product decisions.
- **A model UI is not a production API.** It is one optional client of the
  prediction contract.
- **CI is not deployment.** CI validates and packages; deployment promotes a
  specific immutable artifact.
- **Service monitoring is not model-performance monitoring.** Request success and
  latency can be known immediately; predictive performance often requires later
  ground truth.
- **More dependencies do not make a template more complete.** Optional layers
  should install only what they use.
- **Generality requires contracts and tested extension points, not many unused
  algorithms.**

---

## 10. What can reasonably be used today

After Phase 7, the repository can be used safely as:

- a reference layout for a small tabular DS project;
- a checklist for leakage-aware EDA and preprocessing;
- a starting point for writing dataset-specific experiments;
- a clean-installable Python package;
- an executable domain-neutral binary-classification baseline;
- a working example of one preprocessing-and-estimator artifact;
- a raw JSON inference smoke path and optional Streamlit client;
- a locally enforced, high-coverage baseline with pull-request CI.
- a local MLflow experiment system for comparing the included baseline and
  tuning strategy without selecting on the test set;
- a signature-backed, quality-gated model packaging example with reproducibility
  artifacts and a generated model card.
- an idempotent, schema-validated raw-to-prepared data example with tamper-evident
  manifests and deterministic profiles;
- a teaching example for stratified random or forward-only temporal evaluation.
- a reproducible local/CI model-build example producing a validated,
  commit-addressable MLflow Model bundle;
- a demonstration that failed builds never become final package directories.
- a typed local prediction service with stable validation and availability
  contracts;
- a non-root, read-only container example that mounts an exact package version;
- a CI example that compares direct MLflow and HTTP predictions before retaining
  the package.
- a locally persistent Prometheus/Grafana reference stack with a provisioned
  service dashboard;
- a privacy-safe bounded metric contract and tested operational alert rules;
- a runbook-backed example of model-down, error-rate, and latency response.

It should not yet be trusted unchanged as:

- a general regression, multiclass, time-series, NLP, or vision template;
- a complete schema-validation framework;
- a centrally governed experiment registry or promotion system;
- a production deployment or registry-promotion template;
- an automated retraining system;
- a self-healing production ML platform or automated statistical-drift system.

---

## 11. Relationship to the extension plan

This assessment contains both the original **“before” findings** and the current
Phase 1 status. `project-extension-plan.md` is the remaining migration roadmap
and eventual definition of done.

Recommended reading order for a future session:

1. read this document to understand what exists and why;
2. read Sections 1–6 of `project-extension-plan.md` for target principles and
   contracts;
3. instantiate a real project and complete its customization checklist;
4. update this assessment as milestones are completed so it does not become a
   stale description of the repository.

Phase 1 proved one clean raw-input-to-prediction path, Phase 2 continuously
enforces it through tests and CI, Phase 3 makes model comparison an explicit
evidence trail, Phase 4 connects that evidence to validated prepared data, and
Phase 5 turns the selected result into an immutable validated package. Phase 6
turns that package into a typed, containerized network service. Phase 7 makes
that service observable without confusing output telemetry with model quality.
Phase 8 completes that product layer. Future work belongs to a concrete
instantiated project's domain, registry, deployment platform, and owners.
