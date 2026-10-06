from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, mean_absolute_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.build_features import build_features, get_feature_columns
from src.mlops.model_registry import register_model_bundle
from src.mlops.experiment_tracking import log_streaming_experiment
from src.models.ai4i_metrics import classification_metrics, select_fbeta_threshold

RAW_DATA = Path("data/raw/telemetry.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")


def add_rul_target(df: pd.DataFrame, max_rul_hours: int = 720) -> pd.DataFrame:
    data = df.copy().sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    rul = np.full(len(data), np.nan, dtype=float)

    for _, idx in data.groupby("machine_id", sort=False).groups.items():
        positions = list(idx)
        next_failure: int | None = None
        for pos in reversed(positions):
            if int(data.at[pos, "failure"]) == 1:
                next_failure = pos
                rul[pos] = 0.0
            elif next_failure is not None:
                hours = next_failure - pos
                rul[pos] = float(min(hours, max_rul_hours))

    data["remaining_useful_life_hours"] = rul
    return data


def make_numeric_pipeline(feature_columns: list[str], scale: bool = False) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    preprocessor = ColumnTransformer(
        [("numeric", Pipeline(steps), feature_columns)], remainder="drop"
    )
    return preprocessor


def train_failure_model(features: pd.DataFrame, feature_cols: list[str]) -> tuple[Pipeline, dict]:
    times = pd.to_datetime(features["timestamp"])
    train_cutoff = times.quantile(0.70)
    validation_cutoff = times.quantile(0.85)
    train_df = features[times < train_cutoff]
    validation_df = features[(times >= train_cutoff) & (times < validation_cutoff)]
    test_df = features[times >= validation_cutoff]

    if len(train_df) > 80_000:
        train_df = train_df.sample(80_000, random_state=42)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.5,
        class_weight="balanced",
        random_state=42,
    )
    pipeline = Pipeline(
        [("preprocess", make_numeric_pipeline(feature_cols)), ("model", model)]
    )
    pipeline.fit(train_df[feature_cols], train_df["failure_within_24h"])

    validation_probs = pipeline.predict_proba(validation_df[feature_cols])[:, 1]
    threshold = select_fbeta_threshold(
        validation_df["failure_within_24h"].to_numpy(), validation_probs, beta=2.0
    )
    test_probs = pipeline.predict_proba(test_df[feature_cols])[:, 1]
    evaluated = classification_metrics(
        test_df["failure_within_24h"].to_numpy(), test_probs, threshold
    )
    metrics = {
        **evaluated,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(validation_df)),
        "test_rows": int(len(test_df)),
        "feature_columns": feature_cols,
        "threshold_policy": "maximize validation F2 (beta=2)",
    }
    return pipeline, metrics


def train_anomaly_model(features: pd.DataFrame, feature_cols: list[str]) -> tuple[Pipeline, dict]:
    healthy = features[features["failure_within_24h"] == 0]
    if len(healthy) > 30_000:
        healthy = healthy.sample(30_000, random_state=42)

    pipeline = Pipeline(
        [
            ("preprocess", make_numeric_pipeline(feature_cols, scale=True)),
            (
                "model",
                IsolationForest(
                    n_estimators=140,
                    contamination=0.03,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    pipeline.fit(healthy[feature_cols])
    raw_scores = -pipeline.decision_function(healthy[feature_cols])
    median = float(np.median(raw_scores))
    q25, q75 = np.quantile(raw_scores, [0.25, 0.75])
    scale = float(max(q75 - q25, 1e-6))
    metadata = {
        "score_median": median,
        "score_iqr": scale,
        "feature_columns": feature_cols,
        "healthy_rows": int(len(healthy)),
    }
    return pipeline, metadata


def train_rul_model(raw: pd.DataFrame, features: pd.DataFrame, feature_cols: list[str]) -> tuple[Pipeline, dict]:
    targets = add_rul_target(raw)[["machine_id", "timestamp", "remaining_useful_life_hours"]]
    targets["timestamp"] = pd.to_datetime(targets["timestamp"])
    enriched = features.copy()
    enriched["timestamp"] = pd.to_datetime(enriched["timestamp"])
    enriched = enriched.merge(targets, on=["machine_id", "timestamp"], how="left")
    known = enriched.dropna(subset=["remaining_useful_life_hours"]).copy()

    if len(known) < 100:
        raise RuntimeError("Not enough known run-to-failure rows to train the RUL model.")

    cutoff = known["timestamp"].quantile(0.80)
    train_df = known[known["timestamp"] < cutoff]
    test_df = known[known["timestamp"] >= cutoff]

    if len(train_df) > 80_000:
        train_df = train_df.sample(80_000, random_state=42)
    model = HistGradientBoostingRegressor(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        l2_regularization=0.5,
        random_state=42,
    )
    pipeline = Pipeline(
        [("preprocess", make_numeric_pipeline(feature_cols)), ("model", model)]
    )
    pipeline.fit(train_df[feature_cols], train_df["remaining_useful_life_hours"])
    predictions = pipeline.predict(test_df[feature_cols])
    metadata = {
        "mae_hours": float(mean_absolute_error(test_df["remaining_useful_life_hours"], predictions)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "max_rul_hours": 720,
        "feature_columns": feature_cols,
    }
    return pipeline, metadata


def build_drift_baseline(features: pd.DataFrame, feature_cols: list[str]) -> dict:
    baseline: dict[str, dict] = {}
    for column in feature_cols:
        values = pd.to_numeric(features[column], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) == 0:
            continue
        quantiles = np.unique(np.quantile(values, np.linspace(0, 1, 11)))
        if len(quantiles) < 3:
            quantiles = np.array([values.min() - 1e-6, np.median(values), values.max() + 1e-6])
        counts, edges = np.histogram(values, bins=quantiles)
        proportions = (counts / max(counts.sum(), 1)).tolist()
        baseline[column] = {
            "edges": edges.tolist(),
            "proportions": proportions,
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return baseline


def main() -> None:
    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"{RAW_DATA} not found. Run: python -m src.data.generate_synthetic"
        )

    raw = pd.read_csv(RAW_DATA)
    features = build_features(raw)
    feature_cols = get_feature_columns(features)

    failure_model, failure_meta = train_failure_model(features, feature_cols)
    anomaly_model, anomaly_meta = train_anomaly_model(features, feature_cols)
    rul_model, rul_meta = train_rul_model(raw, features, feature_cols)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(failure_model, MODEL_DIR / "failure_model.joblib")
    joblib.dump(anomaly_model, MODEL_DIR / "anomaly_model.joblib")
    joblib.dump(rul_model, MODEL_DIR / "rul_model.joblib")

    metadata = {
        "failure": failure_meta,
        "anomaly": anomaly_meta,
        "rul": rul_meta,
        "feature_columns": feature_cols,
    }
    (MODEL_DIR / "streaming_metadata.json").write_text(json.dumps(metadata, indent=2))
    (MODEL_DIR / "drift_baseline.json").write_text(
        json.dumps(build_drift_baseline(features, feature_cols), indent=2)
    )
    (REPORT_DIR / "streaming_model_metrics.json").write_text(json.dumps(metadata, indent=2))
    register_model_bundle(MODEL_DIR, metadata)
    log_streaming_experiment(metadata, [REPORT_DIR / "streaming_model_metrics.json"])

    print("Trained streaming model bundle")
    print(f"Failure ROC-AUC: {failure_meta['roc_auc']:.4f}")
    print(f"Failure PR-AUC:  {failure_meta['pr_auc']:.4f}")
    print(f"RUL MAE:         {rul_meta['mae_hours']:.2f} hours")


if __name__ == "__main__":
    main()
