from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from src.data.ai4i import TARGET, load_ai4i
from src.features.ai4i_features import build_ai4i_features

REPORT_DIR = Path("reports")
FIGURE_DIR = REPORT_DIR / "figures"


def main(data_path: Path | None = None) -> None:
    df = build_ai4i_features(load_ai4i(data_path or Path("data/raw/ai4i2020.csv"), download=data_path is None))
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "failure_count": int(df[TARGET].sum()),
        "failure_rate": float(df[TARGET].mean()),
        "type_counts": df["type"].value_counts().to_dict(),
    }
    (REPORT_DIR / "ai4i_eda_summary.json").write_text(json.dumps(summary, indent=2))

    counts = df[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["Healthy", "Failure"], [counts.get(0, 0), counts.get(1, 0)])
    ax.set_title("AI4I class balance")
    ax.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ai4i_class_balance.png", dpi=160)
    plt.close(fig)

    failure_by_type = df.groupby("type", observed=True)[TARGET].mean().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(failure_by_type.index.astype(str), failure_by_type.values)
    ax.set_title("Failure rate by product type")
    ax.set_xlabel("Product type")
    ax.set_ylabel("Failure rate")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "ai4i_failure_rate_by_type.png", dpi=160)
    plt.close(fig)

    numeric = [
        "air_temperature_k","process_temperature_k","rotational_speed_rpm",
        "torque_nm","tool_wear_min","temperature_delta_k",
        "mechanical_power_w","tool_wear_torque",
    ]
    df[numeric + [TARGET]].corr(numeric_only=True).to_csv(REPORT_DIR / "ai4i_correlations.csv")
    print(json.dumps(summary, indent=2))
    print(f"EDA artifacts saved under {REPORT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path, default=None)
    args = parser.parse_args()
    main(args.data_path)
