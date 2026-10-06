from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.data.ai4i import TARGET
from src.features.ai4i_features import get_ai4i_feature_columns

MODEL_PATH = Path("models/ai4i_best_model.joblib")
TEST_PATH = Path("data/processed/ai4i_test.csv")
OUT_PATH = Path("reports/ai4i_permutation_importance.csv")


def main() -> None:
    if not MODEL_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Train the AI4I benchmark first: python -m src.models.train_ai4i")
    pipeline = joblib.load(MODEL_PATH)
    test_df = pd.read_csv(TEST_PATH)
    features = get_ai4i_feature_columns(test_df)
    X_test = test_df[features]
    y_test = test_df[TARGET]
    result = permutation_importance(pipeline, X_test, y_test, n_repeats=10, scoring="average_precision", n_jobs=-1, random_state=42)
    importance = pd.DataFrame({"feature": features,"importance_mean": result.importances_mean,"importance_std": result.importances_std}).sort_values("importance_mean", ascending=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    importance.to_csv(OUT_PATH, index=False)
    print(importance.to_string(index=False))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
