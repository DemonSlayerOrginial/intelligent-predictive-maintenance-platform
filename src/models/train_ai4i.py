from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data.ai4i import TARGET, load_ai4i
from src.features.ai4i_features import assert_no_target_leakage, build_ai4i_features, get_ai4i_feature_columns
from src.models.ai4i_metrics import classification_metrics, select_fbeta_threshold

RANDOM_STATE = 42
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
PROCESSED_DIR = Path("data/processed")


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train, temp = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=RANDOM_STATE)
    validation, test = train_test_split(temp, test_size=0.50, stratify=temp[TARGET], random_state=RANDOM_STATE)
    return train, validation, test


def make_preprocessor(feature_columns: list[str]) -> ColumnTransformer:
    categorical = ["type"]
    numeric = [column for column in feature_columns if column not in categorical]
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ],
        remainder="drop",
    )


def build_models(positive_weight: float) -> dict[str, object]:
    return {
        "logistic_regression": LogisticRegression(class_weight="balanced", max_iter=2500, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=350,max_depth=12,min_samples_leaf=2,
            class_weight="balanced_subsample",n_jobs=-1,random_state=RANDOM_STATE,
        ),
        "xgboost": XGBClassifier(
            n_estimators=350,max_depth=5,learning_rate=0.05,subsample=0.90,
            colsample_bytree=0.90,min_child_weight=2,reg_lambda=1.0,
            scale_pos_weight=positive_weight,eval_metric="logloss",n_jobs=2,random_state=RANDOM_STATE,
        ),
    }


def main(data_path: Path | None = None) -> None:
    raw = load_ai4i(data_path or Path("data/raw/ai4i2020.csv"), download=data_path is None)
    data = build_ai4i_features(raw)
    feature_columns = get_ai4i_feature_columns(data)
    assert_no_target_leakage(feature_columns)
    train_df, validation_df, test_df = split_data(data)
    X_train, y_train = train_df[feature_columns], train_df[TARGET]
    X_validation, y_validation = validation_df[feature_columns], validation_df[TARGET]
    X_test, y_test = test_df[feature_columns], test_df[TARGET]
    positive_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    models = build_models(positive_weight)
    comparison_rows = []
    trained = {}

    for model_name, estimator in models.items():
        pipeline = Pipeline(steps=[("preprocess", make_preprocessor(feature_columns)),("model", estimator)])
        pipeline.fit(X_train, y_train)
        validation_probabilities = pipeline.predict_proba(X_validation)[:, 1]
        threshold = select_fbeta_threshold(y_validation.to_numpy(), validation_probabilities, beta=2.0)
        validation_metrics = classification_metrics(y_validation.to_numpy(), validation_probabilities, threshold)
        test_probabilities = pipeline.predict_proba(X_test)[:, 1]
        test_metrics = classification_metrics(y_test.to_numpy(), test_probabilities, threshold)
        comparison_rows.append({
            "model": model_name,
            "threshold": threshold,
            **{f"val_{k}": v for k, v in validation_metrics.items() if k != "confusion_matrix"},
            **{f"test_{k}": v for k, v in test_metrics.items() if k != "confusion_matrix"},
            "test_confusion_matrix": json.dumps(test_metrics["confusion_matrix"]),
        })
        trained[model_name] = pipeline

    comparison = pd.DataFrame(comparison_rows).sort_values(["val_pr_auc", "val_f2"], ascending=False)
    best_name = str(comparison.iloc[0]["model"])
    best_pipeline = trained[best_name]
    best_threshold = float(comparison.iloc[0]["threshold"])

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_DIR / "ai4i_best_model.joblib")
    (MODEL_DIR / "ai4i_metadata.json").write_text(json.dumps({
        "best_model": best_name,
        "decision_threshold": best_threshold,
        "target": TARGET,
        "feature_columns": feature_columns,
        "selection_metric": "validation PR-AUC, tie-broken by validation F2",
        "threshold_policy": "maximize validation F2 (beta=2)",
        "rows": {"train": len(train_df),"validation": len(validation_df),"test": len(test_df)},
        "train_positive_rate": float(y_train.mean()),
    },indent=2))
    comparison.to_csv(REPORT_DIR / "ai4i_model_comparison.csv", index=False)
    test_df.to_csv(PROCESSED_DIR / "ai4i_test.csv", index=False)

    print("\nAI4I model comparison")
    display_columns = ["model","val_pr_auc","test_pr_auc","test_roc_auc","test_precision","test_recall","test_f2","threshold"]
    print(comparison[display_columns].to_string(index=False))
    print(f"\nSelected model: {best_name}")
    print(f"Saved: {MODEL_DIR / 'ai4i_best_model.joblib'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=None)
    args = parser.parse_args()
    main(args.data_path)
