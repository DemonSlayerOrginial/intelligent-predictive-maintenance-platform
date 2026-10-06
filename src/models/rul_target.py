def add_rul(df):
    df=df.copy()
    out=[]
    for _,g in df.groupby("machine_id"):
        d=[None]*len(g); nxt=None
        for i in range(len(g)-1,-1,-1):
            if g.iloc[i]["observed_failure"]: nxt=0
            d[i]=nxt
            if nxt is not None: nxt+=1
        out.extend(d)
    df["rul_hours"]=out
    return df
