import argparse
from pathlib import Path
import pandas as pd
from src.streaming.simulation import FleetSimulator

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--machines",type=int,default=60)
    p.add_argument("--days",type=int,default=90)
    p.add_argument("--output",default="data/raw/telemetry.csv")
    a=p.parse_args()
    sim=FleetSimulator(a.machines)
    rows=[sim.next_event() for _ in range(a.machines*a.days*24)]
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    print(f"Wrote {len(rows):,} telemetry rows to {out}")

if __name__=="__main__":
    main()
