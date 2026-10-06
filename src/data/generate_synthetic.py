from pathlib import Path
import argparse
import numpy as np
import pandas as pd


def generate_telemetry(num_machines: int = 60, days: int = 90, seed: int = 42) -> pd.DataFrame:
    """Generate realistic-enough telemetry for an ML engineering demo.

    Each machine has a latent wear state. Wear increases over time and accelerates
    with load. Maintenance resets wear. Failure events become increasingly likely
    as wear, temperature, and vibration rise.
    """
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01", periods=days * 24, freq="h")
    rows = []

    for machine_idx in range(num_machines):
        machine_id = f"M{machine_idx:03d}"
        wear = rng.uniform(0.05, 0.25)
        hours_since_maintenance = int(rng.integers(0, 240))
        machine_bias = rng.normal(0, 1)

        for ts in timestamps:
            load = float(np.clip(rng.normal(0.62, 0.16), 0.15, 1.0))
            hours_since_maintenance += 1
            wear += 0.00035 + 0.0008 * load + rng.normal(0, 0.00015)
            wear = max(wear, 0.0)
            temperature = 58 + 31 * wear + 9 * load + machine_bias + rng.normal(0, 1.8)
            vibration = 1.1 + 5.5 * wear + 2.0 * load + rng.normal(0, 0.35)
            pressure = 48 - 8.5 * wear - 2.0 * load + rng.normal(0, 0.9)
            rpm = 1400 + 1250 * load - 170 * wear + rng.normal(0, 45)
            voltage = 230 - 5.0 * wear + rng.normal(0, 1.5)
            error_count = int(rng.poisson(max(0.02, wear * 1.8)))
            risk_logit = (-9.4 + 8.2 * wear + 0.055 * max(temperature - 78, 0) + 0.50 * max(vibration - 4.5, 0) + 0.13 * error_count)
            failure_prob = 1 / (1 + np.exp(-risk_logit))
            failed = int(rng.random() < failure_prob)
            rows.append({"timestamp": ts,"machine_id": machine_id,"temperature": round(float(temperature), 3),"vibration": round(float(vibration), 3),"pressure": round(float(pressure), 3),"rpm": round(float(rpm), 3),"voltage": round(float(voltage), 3),"load": round(float(load), 4),"error_count": error_count,"hours_since_maintenance": hours_since_maintenance,"failure": failed})
            if failed:
                wear = rng.uniform(0.05, 0.16); hours_since_maintenance = 0
            elif hours_since_maintenance > rng.integers(550, 900) and rng.random() < 0.006:
                wear = max(0.04, wear * rng.uniform(0.25, 0.45)); hours_since_maintenance = 0

    df = pd.DataFrame(rows).sort_values(["machine_id", "timestamp"]).reset_index(drop=True)
    def future_failure_window(series: pd.Series) -> pd.Series:
        future_steps = [series.shift(-step) for step in range(1, 25)]
        return pd.concat(future_steps, axis=1).max(axis=1).fillna(0).astype(int)
    df["failure_within_24h"] = df.groupby("machine_id")["failure"].transform(future_failure_window).astype(int)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machines", type=int, default=60)
    parser.add_argument("--days", type=int, default=90)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("data/raw/telemetry.csv"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df = generate_telemetry(args.machines, args.days, args.seed)
    df.to_csv(args.output, index=False)
    rate = df["failure_within_24h"].mean()
    print(f"Wrote {len(df):,} telemetry rows to {args.output}")
    print(f"Positive target rate: {rate:.2%}")


if __name__ == "__main__":
    main()
