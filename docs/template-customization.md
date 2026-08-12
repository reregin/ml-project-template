# Template Customization and Track Guide

## Choose a stopping point

The template is one modular project with three supported stopping points. Start
with the smallest one that meets the project outcome; later tracks extend the
same code rather than replacing it.

### DS-only track

Use this for exploration, leakage-safe preparation, baseline training, and local
inference. Docker, MLflow, FastAPI, Prometheus, and Grafana are not required.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
python -m ml_project_template.example_data
python -m ml_project_template.prepare_data
python -m ml_project_template.train
python -m ml_project_template.predict `
  --input-file examples/binary_classification/sample_request.json
```

Keep `notebooks/`, core package modules, `tests/unit`, and the quality workflow.
You may remove `Dockerfile`, `compose.yaml`, `monitoring/`, serving scripts,
`service.py`, `monitoring.py`, and the package workflow after also removing their
imports, optional dependencies, entry points, tests, docs, and CI references.

### ML-engineering track

Use this when experiments must become tested, comparable, reproducible model
packages, but no network service is required.

```powershell
python -m pip install -e ".[experiment,dev]"
python -m ml_project_template.example_data
python -m ml_project_template.prepare_data
python -m ml_project_template.experiment
python -m ml_project_template.model_package --version local-check
ml-template-validate-package --package artifacts/model-packages/local-check
PowerShell -NoProfile -ExecutionPolicy Bypass -File scripts/check.ps1
```

Keep MLflow, data/package workflows, lineage, model cards, package validation,
and quality gates. The serving and monitoring files listed above remain optional.

### Full MLOps reference track

Use this when the project needs the complete local lifecycle through observable
serving. Complete the ML-engineering commands, copy `.env.example` to ignored
`.env`, set the exact package path and a non-default Grafana password, then run:

```powershell
docker compose config --quiet
docker compose up --build
python scripts/wait_for_service.py --timeout 90
python scripts/smoke_service.py --mode normal
python scripts/smoke_service.py --mode invalid
python scripts/smoke_service.py --mode burst --count 20
python scripts/check_metrics.py
```

Open the API on port 8000, Prometheus on 9090, and Grafana on 3000 unless the
local `.env` overrides them. This is a local reference stack, not automatic
cloud deployment or registry promotion.

<!-- TEMPLATE_MAINTAINER_ONLY_START -->
## Instantiate a renamed project

Run from the template repository and choose a new, nonexistent destination:

```powershell
python scripts/instantiate_template.py `
  --destination D:\projects\example-risk-model `
  --project-name example-risk-model `
  --package-name example_risk_model
```

The command replaces the template title, distribution slug, and Python import
name in text files and renames `src/ml_project_template/`. It deliberately does
not infer domain schema, owners, registries, alert thresholds, or deployment
policy. Review every item below before treating the result as a real project.

A generator framework is deferred. The current structure has one tested set of
identity substitutions and no conditional file-rendering need; introducing
Copier/Cookiecutter now would add a second templating language without improving
the supported tracks. Reconsider when multiple genuinely different generated
variants need conditional prompts or upgrades.
<!-- TEMPLATE_MAINTAINER_ONLY_END -->

## Required customization checklist

### Identity and ownership

- [ ] Project name and package name have been chosen deliberately.
- [ ] Repository/distribution slug and human title are correct.
- [ ] Python package imports, console commands, coverage target, and workflow
  paths use the new package name.
- [ ] `template-manifest.json` has the intended project version or is removed
  after customization.
- [ ] Service, data, model, platform, and incident owners are assigned privately.
- [ ] Escalation and notification contact points are configured outside Git.

### ML problem and data

- [ ] Replace the synthetic generator with authoritative ingestion.
- [ ] State task type: binary, multiclass, regression, ranking, forecasting, or
  another explicitly supported contract.
- [ ] Replace target, numeric/categorical features, accepted types, missingness,
  ranges, categories, minimum rows, and extra-column policy.
- [ ] Choose random, temporal, grouped, or another leakage-safe split matching
  the real prediction moment.
- [ ] Replace the estimator, search space, task metrics, quality threshold, and
  model-card intended-use/limitation evidence.
- [ ] Decide durable raw/prepared data versioning and retention.

### Service and operations

- [ ] Version request/response schemas and define compatibility policy.
- [ ] Set body size, ports, timeouts, resources, replica count, and model memory.
- [ ] Select artifact/model registry, container registry, image naming, promotion
  environments, approvals, and rollback identity.
- [ ] Set alert thresholds from service objectives and representative traffic.
- [ ] Define log retention, Prometheus retention, dashboard ownership, and metric
  network access.
- [ ] Design privacy-reviewed data quality, feature/output drift, and delayed-
  ground-truth performance pipelines separately from request metrics.

### Security and release

- [ ] Replace all `.env.example` markers; never commit the resulting `.env`.
- [ ] Review third-party action SHAs, container image versions/digests, dependency
  audit policy, vulnerability exceptions, SBOM retention, and artifact access.
- [ ] Reassess or remove every entry in `docs/security-exceptions.md` and assign
  its real owner before the recorded deadline.
- [ ] Decide semantic-version compatibility promises and release approvers.
- [ ] Verify every generated-project workflow uses only its own repository.

<!-- TEMPLATE_MAINTAINER_ONLY_START -->
## Detect remaining markers

`template-manifest.json` is the machine-readable inventory of intentional
identity tokens and unresolved customization markers. Inspect occurrences:

```powershell
python scripts/check_template_markers.py
```

After full customization, strict mode should pass:

```powershell
python scripts/check_template_markers.py --strict
```

Feature names such as `numeric_feature_1` are neutral executable examples rather
than magic TODO strings; the checklist—not blind replacement—governs their
domain-specific redesign.
<!-- TEMPLATE_MAINTAINER_ONLY_END -->

## Remove an optional layer safely

Do not delete only the visible directory. Use this sequence:

1. identify its source modules, dependency group, CLI entry points, scripts,
   Compose services, workflow steps, tests, and documentation;
2. remove consumers before providers;
3. update `requirements.txt` and `pyproject.toml` without duplicating authority;
4. run `rg` for removed module/tool names;
5. run the full quality gate from a clean environment;
6. update the README, handbook, assessment, and changelog.

The DS track must remain installable with `pip install -e .`; optional tooling
must never leak into core imports.
