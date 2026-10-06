import pandas as pd

def add_rul(df):
    df=df.copy()
    df["rul_hours"]=df.groupby("machine_id").cumcount(ascending=False)
    return df
