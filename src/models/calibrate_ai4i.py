from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from src.data.ai4i import TARGET


def main() -> None:
    model_path = Path("models/ai4i_best_model.joblib")
    metadata_path = Path("models/ai4i_metadata.json")
    test_path = Path("data/processed/ai4i_test.csv")
    if not (model_path.exists() and metadata_path.exists() and test_path.exists()):
        raise FileNotFoundError("Train the AI4I benchmark before running calibration analysis.")
    model = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    test = pd.read_csv(test_path)
    y = test[TARGET]
    probs = model.predict_proba(test[metadata["feature_columns"]])[:, 1]
    observed, predicted = calibration_curve(y, probs, n_bins=10, strategy="quantile")
    report_dir = Path("reports")
    figure_dir = report_dir / "figures"
    report_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True)
    pd.DataFrame({"mean_predicted_probability": predicted, "observed_failure_rate": observed}).to_csv(report_dir / "ai4i_calibration.csv", index=False)
    (report_dir / "ai4i_calibration_metrics.json").write_text(json.dumps({"brier_score": float(brier_score_loss(y, probs))}, indent=2))
    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "--")
    plt.plot(predicted, observed, marker="o")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed failure rate")
    plt.title("AI4I calibration curve")
    plt.tight_layout()
    plt.savefig(figure_dir / "ai4i_calibration.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
