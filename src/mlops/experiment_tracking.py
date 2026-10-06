from __future__ import annotations

from pathlib import Path


def log_streaming_experiment(metadata: dict, artifact_paths: list[Path] | None = None) -> bool:
    try:
        import mlflow
    except ImportError:
        return False
    mlflow.set_experiment("predictive-maintenance-streaming")
    with mlflow.start_run():
        failure = metadata.get("failure", {})
        rul = metadata.get("rul", {})
        numeric_metrics = {
            "failure_roc_auc": failure.get("roc_auc"),
            "failure_pr_auc": failure.get("pr_auc"),
            "failure_precision": failure.get("precision"),
            "failure_recall": failure.get("recall"),
            "failure_f2": failure.get("f2"),
            "rul_mae_hours": rul.get("mae_hours"),
        }
        mlflow.log_metrics({k: float(v) for k, v in numeric_metrics.items() if v is not None})
        mlflow.log_param("failure_threshold_policy", failure.get("threshold_policy", "unknown"))
        mlflow.log_dict(metadata, "streaming_metadata.json")
        for path in artifact_paths or []:
            if path.exists():
                mlflow.log_artifact(str(path))
    return True
