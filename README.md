# Intelligent Predictive Maintenance Platform

End-to-end ML engineering system for real-time equipment health monitoring.

Features: failure-risk prediction, anomaly detection, remaining useful life estimation, Kafka streaming, online rolling features, FastAPI inference, PostgreSQL history, Redis latest-state caching, WebSocket updates, a React dashboard, drift monitoring, retraining controls, Docker, Kubernetes, Prometheus, and Grafana.

The project uses three separate ML models: a supervised failure classifier, an Isolation Forest anomaly detector, and an RUL regressor. Reference synthetic-data results use chronological holdouts: ROC-AUC ~0.71 and RUL MAE ~61 hours.

Generated datasets and serialized model binaries are intentionally excluded so models are trained and loaded in the same environment.

This is a portfolio/development platform, not a safety-certified industrial control system.
