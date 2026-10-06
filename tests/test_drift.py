import json
from pathlib import Path
import pandas as pd
from src.mlops.drift import compute_feature_drift, population_stability_index


def test_psi_zero_for_identical_distribution():
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_drift_detects_distribution_shift(tmp_path: Path):
    baseline = {"temperature": {"edges": [0, 1, 2],"proportions": [0.5, 0.5],"mean": 1.0,"std": 0.5}}
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps(baseline))
    recent = pd.DataFrame({"temperature": [1.8] * 100})
    report = compute_feature_drift(recent, path)
    assert report["features_drifted"] == 1
    assert report["features"]["temperature"]["status"] == "drift"
