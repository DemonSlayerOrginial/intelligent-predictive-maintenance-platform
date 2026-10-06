# Interview walkthrough

## 1. Why this project exists

The goal is to demonstrate the entire ML-system lifecycle rather than optimize one notebook metric. The model is one component in a system that must ingest events, preserve per-machine ordering, compute online features, serve predictions, store history, alert operators, detect drift, and safely manage model updates.

## 2. Leakage

The forward-looking target is derived from future failures, but online input features use only the current and prior observations. AI4I failure-mode flags are blocked entirely because they reveal target information.

## 3. Why PR-AUC matters

Failures are rare. Accuracy can therefore look excellent even if the classifier misses nearly every failure. PR-AUC and recall-focused threshold selection are more informative. Thresholds are selected on validation data rather than the test set.

## 4. Online/offline feature parity

The simulator advances one logical hour per fleet cycle. The stream processor keeps 24 observations per machine and calculates the same 6h/24h features used during training. In a real system these would use event-time windows and a durable stream processor/state store.

## 5. Kafka keying

Telemetry is keyed by `machine_id`. Kafka partitioning therefore keeps events for one machine ordered within a partition, which is important for rolling features. Multiple stream-processor replicas can consume separate partitions.

## 6. Three ML models, not one renamed three times

- classifier: forward-looking failure probability
- Isolation Forest: unsupervised deviation from healthy behavior
- regressor: remaining useful life

The operational risk state combines them because they answer different questions.

## 7. PostgreSQL vs Redis

PostgreSQL stores durable prediction and alert history for analysis. Redis stores only the latest machine state for low-latency dashboard reads and publishes live updates to the WebSocket backend.

## 8. Drift and delayed labels

Feature drift can be measured immediately, but supervised performance cannot be known until outcomes arrive. The platform therefore generates a retraining request on drift and waits for labeled data. This avoids the common anti-pattern of retraining a classifier on unlabeled production traffic.

## 9. Model promotion

Retraining produces a candidate first. The current implementation promotes only if candidate holdout PR-AUC is at least as good as the champion. A production version would add minimum recall constraints, calibration, shadow/canary deployment, rollback, and human approval.

## 10. Scaling

Inference is stateless and horizontally scalable. Stream processors scale by Kafka partitions. PostgreSQL would normally be managed with replicas/partitioning; Redis can become a cluster. For stream processors, consumer lag is a better autoscaling signal than CPU, so the Kubernetes README recommends KEDA for a production version.
