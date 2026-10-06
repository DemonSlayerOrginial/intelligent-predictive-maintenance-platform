from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from src.features.build_features import get_feature_columns
from src.mlops.model_registry import register_model_bundle
from src.models.train_streaming_models import train_failure_model


def main(labeled_path: Path) -> None:
    if not labeled_path.exists():
        raise FileNotFoundError(
            f"{labeled_path} does not exist. Supervised retraining requires delayed ground-truth labels."
        )
    data = pd.read_csv(labeled_path)
    required = {"timestamp", "machine_id", "failure", "failure_within_24h"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Labeled retraining data is missing columns: {sorted(missing)}")

    feature_cols = get_feature_columns(data)
    model, metrics = train_failure_model(data, feature_cols)
    current_metrics_path = Path("models/streaming_metadata.json")
    current = json.loads(current_metrics_path.read_text()) if current_metrics_path.exists() else {}
    current_pr_auc = float(current.get("failure", {}).get("pr_auc", -1))

    candidate_dir = Path("models/candidates")
    candidate_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = candidate_dir / "failure_model.joblib"
    joblib.dump(model, candidate_path)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_metrics": metrics,
        "current_pr_auc": current_pr_auc,
        "promoted": False,
    }

    if metrics["pr_auc"] >= current_pr_auc:
        joblib.dump(model, Path("models/failure_model.joblib"))
        current["failure"] = metrics
        Path("models/streaming_metadata.json").write_text(json.dumps(current, indent=2))
        register_model_bundle(Path("models"), current)
        report["promoted"] = True

    Path("reports").mkdir(exist_ok=True)
    Path("reports/retraining_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labeled-path", type=Path, required=True)
    args = parser.parse_args()
    main(args.labeled_path)
