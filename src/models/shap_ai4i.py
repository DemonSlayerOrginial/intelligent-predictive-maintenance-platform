from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install requirements-mlops.txt to run SHAP analysis.") from exc

    model_path = Path("models/ai4i_best_model.joblib")
    metadata_path = Path("models/ai4i_metadata.json")
    test_path = Path("data/processed/ai4i_test.csv")
    if not (model_path.exists() and metadata_path.exists() and test_path.exists()):
        raise FileNotFoundError("Train the AI4I benchmark before running SHAP analysis.")
    pipeline = joblib.load(model_path)
    metadata = json.loads(metadata_path.read_text())
    test = pd.read_csv(test_path)
    test = test.sample(min(250, len(test)), random_state=42)
    X = test[metadata["feature_columns"]]
    preprocessor = pipeline.named_steps["preprocess"]
    estimator = pipeline.named_steps["model"]
    transformed = preprocessor.transform(X)
    names = preprocessor.get_feature_names_out()
    explainer = shap.Explainer(estimator, transformed)
    values = explainer(transformed)
    if getattr(values, "values", None) is not None and values.values.ndim == 3:
        values = values[:, :, 1]
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    shap.summary_plot(values, transformed, feature_names=names, show=False)
    plt.tight_layout()
    plt.savefig(out / "ai4i_shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()
