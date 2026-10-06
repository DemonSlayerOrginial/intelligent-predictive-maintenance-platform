from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import pandas as pd

AI4I_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv"
DEFAULT_PATH = Path("data/raw/ai4i2020.csv")

COLUMN_RENAME = {
    "UDI": "udi",
    "Product ID": "product_id",
    "Type": "type",
    "Air temperature [K]": "air_temperature_k",
    "Process temperature [K]": "process_temperature_k",
    "Rotational speed [rpm]": "rotational_speed_rpm",
    "Torque [Nm]": "torque_nm",
    "Tool wear [min]": "tool_wear_min",
    "Machine failure": "machine_failure",
    "TWF": "twf",
    "HDF": "hdf",
    "PWF": "pwf",
    "OSF": "osf",
    "RNF": "rnf",
}

LEAKAGE_COLUMNS = ["twf", "hdf", "pwf", "osf", "rnf"]
ID_COLUMNS = ["udi", "product_id"]
TARGET = "machine_failure"
BASE_FEATURES = [
    "type",
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]


def download_ai4i(path: Path = DEFAULT_PATH, force: bool = False) -> Path:
    path = Path(path)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(AI4I_URL, path)
    return path


def load_ai4i(path: Path = DEFAULT_PATH, download: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        if not download:
            raise FileNotFoundError(f"{path} does not exist")
        download_ai4i(path)
    df = pd.read_csv(path).rename(columns=COLUMN_RENAME)
    missing = set(COLUMN_RENAME.values()) - set(df.columns)
    if missing:
        raise ValueError(f"AI4I file is missing expected columns: {sorted(missing)}")
    return df
