from __future__ import annotations
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline

from src.features.build_features import build_features, get_feature_columns

RAW_DATA = Path("data/raw/telemetry.csv")
PROCESSED_DATA = Path("data/processed/features.csv")
MODEL_PATH = Path("models/failure_model.joblib")
FEATURES_PATH = Path("models/feature_columns.json")
METRICS_PATH = Path("models/metrics.json")


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(f"{RAW_DATA} not found. Run: python -m src.data.generate_synthetic")

    raw = pd.read_csv(RAW_DATA)
    features = build_features(raw)
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(PROCESSED_DATA, index=False)

    feature_cols = get_feature_columns(features)
    cutoff = pd.to_datetime(features["timestamp"]).quantile(0.80)
    train_df = features[pd.to_datetime(features["timestamp"]) < cutoff]
    test_df = features[pd.to_datetime(features["timestamp"]) >= cutoff]
    X_train, y_train = train_df[feature_cols], train_df["failure_within_24h"]
    X_test, y_test = test_df[feature_cols], test_df["failure_within_24h"]

    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    preprocessor = ColumnTransformer([("numeric", numeric_pipeline, feature_cols)], remainder="drop")
    model = RandomForestClassifier(
        n_estimators=250,max_depth=14,min_samples_leaf=3,
        class_weight="balanced_subsample",n_jobs=-1,random_state=42,
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)
    metrics = {
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "positive_rate_train": float(y_train.mean()),
        "positive_rate_test": float(y_test.mean()),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "pr_auc": float(average_precision_score(y_test, probabilities)),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        "threshold": 0.50,
        "time_split_cutoff": str(cutoff),
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    FEATURES_PATH.write_text(json.dumps(feature_cols, indent=2))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(f"Saved model to {MODEL_PATH}")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:  {metrics['pr_auc']:.4f}")
    print("Confusion matrix:", metrics["confusion_matrix"])


if __name__ == "__main__":
    main()
