"""Print an API-ready prediction payload from the newest engineered feature row."""
import json
import pandas as pd
from src.features.build_features import get_feature_columns

features = pd.read_csv("data/processed/features.csv")
row = features.iloc[-1]
columns = get_feature_columns(features)
payload = {
    "machine_id": row["machine_id"],
    "features": {name: float(row[name]) for name in columns},
}
print(json.dumps(payload, indent=2))
