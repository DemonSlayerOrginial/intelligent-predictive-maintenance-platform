"""Generate an offline, schema-compatible AI4I-like fixture for development only.

This is NOT the UCI dataset and must not be used for reported benchmark results.
"""
from pathlib import Path
import numpy as np
import pandas as pd

OUT = Path("data/dev/ai4i_fixture.csv")


def main(rows: int = 6000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    product_type = rng.choice(["L", "M", "H"], size=rows, p=[0.5, 0.3, 0.2])
    air = rng.normal(300.0, 2.0, rows)
    process = air + rng.normal(10.0, 1.0, rows)
    rpm = np.clip(rng.normal(1538, 180, rows), 900, 2900).round().astype(int)
    torque = np.clip(rng.normal(40.0, 10.0, rows), 4, 80)
    tool_wear = rng.integers(0, 241, rows)
    power = torque * rpm * (2 * np.pi / 60.0)
    twf = ((tool_wear >= 210) & (rng.random(rows) < 0.20)).astype(int)
    hdf = (((process - air) < 8.6) & (rpm < 1380)).astype(int)
    pwf = ((power < 3500) | (power > 9000)).astype(int)
    threshold = np.where(product_type == "L", 11000, np.where(product_type == "M", 12000, 13000))
    osf = ((tool_wear * torque) > threshold).astype(int)
    rnf = (rng.random(rows) < 0.001).astype(int)
    failure = np.maximum.reduce([twf, hdf, pwf, osf, rnf])
    product_ids = [f"{t}{10000+i}" for i, t in enumerate(product_type)]
    df = pd.DataFrame({
        "UDI": np.arange(1, rows + 1),"Product ID": product_ids,"Type": product_type,
        "Air temperature [K]": np.round(air, 1),"Process temperature [K]": np.round(process, 1),
        "Rotational speed [rpm]": rpm,"Torque [Nm]": np.round(torque, 1),
        "Tool wear [min]": tool_wear,"Machine failure": failure,
        "TWF": twf,"HDF": hdf,"PWF": pwf,"OSF": osf,"RNF": rnf,
    })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Saved offline dev fixture to {OUT}")
    print(f"Rows: {len(df):,}; failure rate: {df['Machine failure'].mean():.3%}")


if __name__ == "__main__":
    main()
