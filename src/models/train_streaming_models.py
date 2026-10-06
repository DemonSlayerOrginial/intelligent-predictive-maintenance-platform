from pathlib import Path
import json, joblib, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error
from src.features.build_features import build_features, feature_columns

def add_rul(df):
    out=[]
    for _,g in df.groupby("machine_id"):
        nxt=None; vals=[]
        for fail in g["observed_failure"].to_numpy()[::-1]:
            if fail: nxt=0
            vals.append(np.nan if nxt is None else nxt)
            if nxt is not None: nxt+=1
        out.extend(vals[::-1])
    df=df.copy(); df["rul_hours"]=out
    return df

def main():
    df=add_rul(build_features(pd.read_csv("data/raw/telemetry.csv")))
    cols=feature_columns(df)
    n=len(df); cut=int(n*.8)
    tr,te=df.iloc[:cut],df.iloc[cut:]
    X,y=tr[cols],tr["failure_within_24h"]
    clf=HistGradientBoostingClassifier(max_iter=120,max_depth=6).fit(X,y)
    iso=IsolationForest(contamination=.03,random_state=42).fit(X[y==0])
    rr=tr.dropna(subset=["rul_hours"])
    rul=HistGradientBoostingRegressor(max_iter=120,max_depth=6).fit(rr[cols],rr["rul_hours"])
    Path("models").mkdir(exist_ok=True)
    joblib.dump(clf,"models/failure_model.joblib")
    joblib.dump(iso,"models/anomaly_model.joblib")
    joblib.dump(rul,"models/rul_model.joblib")
    Path("models/feature_columns.json").write_text(json.dumps(cols))
    p=clf.predict_proba(te[cols])[:,1]
    rt=te.dropna(subset=["rul_hours"])
    print("ROC-AUC",round(roc_auc_score(te["failure_within_24h"],p),3))
    print("PR-AUC",round(average_precision_score(te["failure_within_24h"],p),3))
    print("RUL MAE",round(mean_absolute_error(rt["rul_hours"],rul.predict(rt[cols])),1))

if __name__=="__main__":
    main()
