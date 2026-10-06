from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def population_stability_index(expected: list[float], actual: list[float], eps: float = 1e-6) -> float:
    if len(expected) != len(actual):
        raise ValueError("Expected and actual distributions must use the same bins")
    e = np.clip(np.asarray(expected, dtype=float), eps, None)
    a = np.clip(np.asarray(actual, dtype=float), eps, None)
    e = e / e.sum()
    a = a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))


def compute_feature_drift(recent: pd.DataFrame, baseline_path: Path = Path("models/drift_baseline.json")) -> dict:
    baseline = json.loads(baseline_path.read_text())
    feature_reports: dict[str, dict] = {}
    drifted = 0
    for feature, stats in baseline.items():
        if feature not in recent.columns:
            continue
        values = pd.to_numeric(recent[feature], errors="coerce").dropna().to_numpy(dtype=float)
        if len(values) < 20:
            continue
        edges = np.asarray(stats["edges"], dtype=float).copy()
        edges[0] = -np.inf
        edges[-1] = np.inf
        counts, _ = np.histogram(values, bins=edges)
        actual = (counts / max(counts.sum(), 1)).tolist()
        psi = population_stability_index(stats["proportions"], actual)
        status = "drift" if psi >= 0.20 else "watch" if psi >= 0.10 else "stable"
        if status == "drift":
            drifted += 1
        feature_reports[feature] = {
            "psi": round(psi, 6),
            "status": status,
            "recent_mean": float(np.mean(values)),
            "baseline_mean": float(stats["mean"]),
        }
    evaluated = len(feature_reports)
    return {
        "features_evaluated": evaluated,
        "features_drifted": drifted,
        "drift_fraction": (drifted / evaluated) if evaluated else 0.0,
        "requires_retraining_review": bool(evaluated and drifted / evaluated >= 0.20),
        "features": feature_reports,
    }
