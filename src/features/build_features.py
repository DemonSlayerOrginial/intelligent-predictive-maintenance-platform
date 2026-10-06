import pandas as pd
SIGNALS=["temperature","vibration","pressure","rpm","voltage","load"]

def build_features(df):
    df=df.copy()
    df["timestamp"]=pd.to_datetime(df["timestamp"],utc=True)
    df=df.sort_values(["machine_id","timestamp"])
    g=df.groupby("machine_id")
    for s in SIGNALS:
        df[f"{s}_mean_6h"]=g[s].transform(lambda x:x.rolling(6,min_periods=6).mean())
        df[f"{s}_std_24h"]=g[s].transform(lambda x:x.rolling(24,min_periods=7).std())
        df[f"{s}_delta_6h"]=g[s].diff(6)
    df["errors_6h"]=g["error_count"].transform(lambda x:x.rolling(6,min_periods=6).sum())
    df["errors_24h"]=g["error_count"].transform(lambda x:x.rolling(24,min_periods=7).sum())
    df["failure_within_24h"]=g["observed_failure"].transform(lambda x:x.shift(-1).iloc[::-1].rolling(24,min_periods=1).max().iloc[::-1])
    return df.dropna().reset_index(drop=True)

def feature_columns(df):
    skip={"timestamp","machine_id","observed_failure","failure_within_24h"}
    return [c for c in df if c not in skip]
