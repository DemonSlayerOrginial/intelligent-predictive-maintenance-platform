# Intelligent Predictive Maintenance Platform

A portfolio-grade end-to-end ML engineering system for predicting industrial equipment risk from streaming telemetry.

This is intentionally **not** a notebook with a model behind a UI. It covers the lifecycle from data generation and benchmark experimentation through real-time inference, persistence, monitoring, drift detection, retraining controls, Docker, and Kubernetes-ready deployment.

## What the system predicts

For each machine, the live inference bundle produces three independent ML outputs:

1. **Failure probability** — probability of a failure within the next 24 logical hours.
2. **Anomaly score** — unsupervised Isolation Forest signal for behavior outside the healthy training distribution.
3. **Remaining Useful Life (RUL)** — estimated hours until the next failure in the synthetic run-to-failure track.

Those signals are combined into a `healthy`, `warning`, or `critical` operational state.

## Architecture

```text
                         OFFLINE ML

Synthetic run-to-failure data ──→ rolling features ──→ failure classifier
                         │                         ├──→ anomaly detector
                         │                         └──→ RUL regressor
                         │
                         └──→ baseline distributions → drift monitor

AI4I UCI benchmark ──→ EDA → model comparison → calibration / SHAP


                         ONLINE SYSTEM

Machine simulator
      │
      ▼
Kafka: machine-telemetry
      │
      ▼
Streaming feature processor ────────┐
  • per-machine rolling state       │
  • 6h / 24h logical windows        ▼
                              Inference API
                              • failure model
                              • anomaly model
                              • RUL model
                                      │
                                      ▼
                             Kafka: predictions
                                      │
                                      ▼
                              Persistence worker
                               /              \
                              ▼                ▼
                         PostgreSQL          Redis
                        history/alerts     latest state
                              │                │
                              └──────┬─────────┘
                                     ▼
                               Backend API
                              REST + WebSocket
                                     │
                                     ▼
                              React dashboard

Prometheus ← inference / stream processor / backend / drift monitor → Grafana
```

The live simulator advances a **logical hourly clock** while publishing faster in wall-clock time. That keeps online 6-event and 24-event windows aligned with the offline 6h/24h feature definitions while making the demo fun to watch.

## Phase status

### Phase 1 — ML foundation ✅

- Synthetic multi-machine telemetry
- Forward-looking `failure_within_24h` target
- Leakage-safe rolling features
- Time-based evaluation
- FastAPI model serving

### Phase 2 — serious ML experimentation ✅

- UCI AI4I 2020 benchmark pipeline
- Logistic Regression, Random Forest, and XGBoost comparison
- PR-AUC / ROC-AUC / precision / recall / F1 / F2
- Validation-only threshold selection
- Leakage guardrails for AI4I failure-mode flags
- Permutation importance
- Calibration analysis
- Optional SHAP explanation script
- Optional MLflow experiment logging

### Phase 3 — real-time streaming system ✅

- Stateful fleet simulator
- Kafka telemetry topic
- Per-machine online feature windows
- Real-time inference calls
- Predictions topic
- Prometheus stream metrics

### Phase 4 — application layer ✅

- PostgreSQL prediction history + alerts
- Redis latest-state cache + pub/sub
- Backend REST API
- WebSocket live updates
- React command-center dashboard

### Phase 5 — advanced ML ✅

- Supervised failure-risk classifier
- Isolation Forest anomaly detection
- Remaining Useful Life regression
- Combined operational risk policy

### Phase 6 — MLOps / deployment ✅

- Versioned local model registry
- Feature-distribution baseline
- PSI drift monitoring
- Retraining-request workflow
- Conservative candidate promotion using labeled holdout PR-AUC
- Prometheus + provisioned Grafana dashboard
- Docker Compose stack
- Kubernetes application manifests + inference HPA
- GitHub Actions CI

## Current streaming-model results

The reference training run used the synthetic run-to-failure generator with a **forward time split**, so later operating periods are intentionally harder than training data.

- Failure classifier ROC-AUC: approximately **0.71**
- Failure classifier PR-AUC: approximately **0.08**
- RUL MAE: approximately **61 hours**

These numbers are not presented as an industrial benchmark. The synthetic generator exists to make the entire online ML system reproducible. AI4I is kept as a separate tabular benchmark, and the architecture is ready for a real run-to-failure dataset such as C-MAPSS as a future dataset swap.

## Fastest way to verify the ML core

Generated datasets and serialized model binaries are intentionally **not committed**. This avoids large repository files and scikit-learn/joblib compatibility problems across environments. Train the bundle locally first:

```bash
pip install -r requirements.txt
python -m src.data.generate_synthetic --machines 60 --days 90
python -m src.models.train_streaming_models
python -m scripts.local_smoke
```

That executes:

```text
simulator → online rolling features → failure/anomaly/RUL inference → risk state
```

without requiring Kafka, PostgreSQL, or Redis.

## Train the streaming models yourself

```bash
python -m src.data.generate_synthetic --machines 60 --days 90
python -m src.models.train_streaming_models
pytest -q
```

Artifacts are written under:

```text
models/
  failure_model.joblib
  anomaly_model.joblib
  rul_model.joblib
  streaming_metadata.json
  drift_baseline.json
  registry/

reports/
  streaming_model_metrics.json
```

If you install the optional MLOps dependencies, training is also logged to local MLflow:

```bash
pip install -r requirements-mlops.txt
python -m src.models.train_streaming_models
mlflow ui
```

## Run the full platform

Train the models first (using the same Python environment that will load them), then:

```bash
python -m src.data.generate_synthetic --machines 60 --days 90
python -m src.models.train_streaming_models
docker compose up --build
```

Open:

| Service | URL |
|---|---|
| Operations dashboard | `http://localhost:3000` |
| Backend API | `http://localhost:8000/docs` |
| Inference API | `http://localhost:8001/docs` |
| Prometheus | `http://localhost:9090` |
| Grafana | `http://localhost:3001` |

Grafana development login: `admin / admin`.

The dashboard waits for roughly 7 logical observations per machine before the first rolling-feature predictions appear.

## Reproducibility and model artifacts

The repository tracks source code, configuration, metrics, and model metadata, but not generated CSV datasets or `*.joblib` binaries. Scikit-learn does not guarantee persisted estimator compatibility across library versions, so the intended workflow is to train and load models in the same environment. The core dependency is pinned to `scikit-learn==1.9.0` for a reproducible demo setup.

## AI4I benchmark

```bash
python -m src.data.download_ai4i
python -m src.data.eda_ai4i
python -m src.models.train_ai4i
python -m src.models.explain_ai4i
python -m src.models.calibrate_ai4i
```

Optional SHAP:

```bash
pip install -r requirements-mlops.txt
python -m src.models.shap_ai4i
```

AI4I source: UCI Machine Learning Repository, DOI `10.24432/C5HS5C`, CC BY 4.0.

### Why AI4I failure-mode flags are blocked

`TWF`, `HDF`, `PWF`, `OSF`, and `RNF` directly describe failure modes associated with the target. Feeding those flags into the classifier would leak information about the answer and create an unrealistic benchmark. The training pipeline explicitly prevents them from becoming inputs.

## Drift and retraining behavior

The drift monitor calculates **Population Stability Index (PSI)** for production features against the training baseline.

- PSI `< 0.10`: stable
- PSI `0.10–0.20`: watch
- PSI `>= 0.20`: drift

If at least 20% of monitored features cross the drift threshold, the service writes `reports/retrain_request.json`.

It does **not** immediately retrain a supervised model from unlabeled live traffic. Once delayed ground-truth labels are available, run:

```bash
python -m src.mlops.retrain --labeled-path data/retraining/labeled_features.csv
```

The candidate only replaces the current failure classifier if its labeled holdout PR-AUC is at least as strong as the current model.

## Repository layout

```text
src/
  data/                  dataset generation, download, EDA
  features/              offline feature engineering
  models/                training, evaluation, inference
  streaming/             simulator state + online features
  mlops/                 drift, registry, retraining, MLflow hook

services/
  simulator/             Kafka telemetry producer
  stream_processor/      Kafka consumer + rolling features
  inference/             FastAPI ML inference
  persistence/           PostgreSQL + Redis writer
  backend/               REST + WebSocket application API
  drift_monitor/         production PSI monitor

dashboard/               React/Vite live operations UI
infra/                    PostgreSQL, Prometheus, Grafana configs
k8s/                      Kubernetes application deployments
scripts/                  local smoke test
.github/workflows/        CI
```

## Interview talking points

See `docs/interview-walkthrough.md` for the design decisions worth discussing: leakage, class imbalance, PR-AUC, online/offline feature parity, Kafka partitioning, idempotency, Redis vs PostgreSQL, anomaly detection, delayed labels, drift, model promotion, and scaling.

## Engineering caveats

This is a portfolio/development platform, not a safety-certified industrial control system. The demo dataset and RUL labels are synthetic. Before real equipment use, the model would need domain-specific sensor validation, calibrated costs/thresholds, reliable delayed labels, safety review, model governance, secure service authentication, schema/version management, and production-grade stateful infrastructure.
