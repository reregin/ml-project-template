# Prediction Service Operations Runbook

## Purpose and ownership

This runbook covers the three Phase 7 operational alerts for the generic
prediction service. Replace the placeholder ownership and escalation details
when instantiating the template. Never put notification tokens, webhook URLs,
credentials, customer data, or raw prediction inputs in this file.

The committed stack evaluates alerts in Prometheus. Production notification
routing should be supplied through an externally managed Alertmanager or the
organization's observability platform. Configure contact points and secrets
outside Git, test them in a non-production environment, and document the owning
team in the instantiated project's private operations system.

## First response checklist

1. Record alert name, start time, environment, and affected service revision.
2. Open the **ML Prediction Service** dashboard and confirm the symptom.
3. Check `/live`, `/ready`, and Prometheus target health independently.
4. Correlate service logs by timestamp; do not copy payloads into tickets.
5. Stop rollouts or roll back only when evidence connects the change to impact.
6. Record mitigation, recovery time, and a follow-up owner.

## Model unavailable

Alert: `MLModelUnavailable` when `ml_model_ready < 1` for 30 seconds.

Meaning: the HTTP process may be alive, but the package did not validate/load or
the service cannot expose predictions. This is availability, not model quality.

Diagnose:

1. Call `/live`; if it fails, inspect container/process scheduling and logs.
2. Call `/ready`; expect `503 model_unavailable` while the alert is active.
3. Find `event=model_load_failed` in logs and note only its exception type.
4. Verify the configured package path exists and is mounted read-only.
5. Run `ml-template-validate-package --package <path>` from a trusted environment.
6. Check package hashes, permissions, runtime dependency compatibility, and free
   memory/disk before restarting.

Mitigate by restoring the last known-good immutable package or service revision.
Do not edit a versioned package in place. Resolve only after readiness remains 1
and a known fixture predicts successfully.

## High prediction error rate

Alert: `MLPredictionErrorRateHigh` when invalid, unavailable, or failed outcomes
exceed 10% of prediction requests for two minutes.

Diagnose:

1. Split the dashboard outcome series into `invalid`, `unavailable`, and `error`.
2. For `invalid`, compare deploy time with client/schema changes and inspect only
   bounded validation types—not raw values.
3. For `unavailable`, follow **Model unavailable** above.
4. For `error`, inspect timestamp-correlated server exceptions, resource limits,
   and dependency failures.
5. Confirm Prometheus is not mistaking a counter reset after restart for errors.

Mitigate client incompatibility with a coordinated schema/client rollback or a
backward-compatible service fix. Mitigate server failures by restoring the last
known-good version. Do not retry invalid `4xx` requests. Resolve after the ratio
stays below threshold for a representative traffic window.

## High prediction latency

Alert: `MLPredictionLatencyHigh` when five-minute p95 end-to-end prediction
duration exceeds one second for two minutes.

Diagnose:

1. Compare request rate, in-progress requests, error ratio, CPU, and memory.
2. Determine whether the change began with a model/package or infrastructure
   revision.
3. Compare direct model inference with HTTP duration using the committed fixture.
4. Check CPU throttling, memory pressure, host contention, dependency stalls, and
   oversized request patterns.
5. Confirm there is enough traffic for a meaningful percentile.

Mitigate saturation by scaling replicas only after checking per-model memory and
CPU. Roll back a slower model when it violates the service objective. Bound
client retries with timeouts, backoff, and jitter so retries do not amplify load.

## Test rules and scenarios

Validate Prometheus configuration and all three synthetic alert cases:

```powershell
docker compose exec prometheus promtool check config /etc/prometheus/prometheus.yml
docker compose exec prometheus `
  promtool test rules /etc/prometheus/alerts.test.yml
```

Generate visible operational traffic after the stack is ready:

```powershell
python scripts/smoke_service.py --mode normal
python scripts/smoke_service.py --mode burst --count 50
python scripts/smoke_service.py --mode invalid
python scripts/check_metrics.py
```

The deterministic `alerts.test.yml` time series test alert expressions and `for`
durations without damaging a running service. For an end-to-end model-down drill,
use an isolated non-production stack with a missing package path, confirm
`/live` remains 200 and `/ready` becomes 503, observe the alert, then restore the
known-good path. Never break a shared or production mount to test an alert.

Thresholds and `for` durations are explicit in
`monitoring/prometheus/alerts.yml`. Tune them from measured service objectives,
traffic volume, and an agreed noise budget; rerun the rule tests after changes.

## Retention and restarts

Prometheus stores samples in the `prometheus-data` named volume and defaults to
seven-day retention through `PROMETHEUS_RETENTION`. Grafana uses `grafana-data`
for its local database, while its datasource and dashboard definitions remain
reproducible files. Normal `docker compose stop`/`start` and service replacement
therefore retain monitoring history. `docker compose down -v` intentionally
deletes both volumes and history.

Application counters reset when the prediction process restarts; Prometheus
counter functions such as `rate()` account for resets, and historical samples
remain in Prometheus. Back up or use remotely managed long-term storage when the
required retention exceeds a single host's lifecycle.

## Statistical model health is separate

Operational metrics answer whether the service is reachable, fast, and returning
valid responses. They do not prove prediction correctness.

- Data quality should validate schema, missingness, ranges, and category policy
  before training and at an approved inference aggregation boundary.
- Feature drift should compare privacy-reviewed aggregate windows against a
  versioned reference distribution outside Prometheus labels.
- Output drift may track bounded class proportions, but a change is only a signal
  for investigation—not proof of degradation.
- True performance requires securely joining predictions to delayed ground truth,
  computing task metrics on representative windows, and accounting for label
  delay and selection bias.

Do not label prediction counters as accuracy, precision, recall, or business
success without ground truth.
