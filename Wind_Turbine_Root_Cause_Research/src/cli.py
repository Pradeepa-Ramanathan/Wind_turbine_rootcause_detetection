import os, json, argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from .care_data import discover_events, load_event
from .preprocessing import prepare_splits
from .graphs import build_graphs
from .model import GraphTemporalForecaster, make_windows
from .localization import residual_stats
from .experiment import forecast_metrics

load_dotenv()
DATA_DIR=os.getenv("CARE_DATA_DIR","")
ARTIFACT_DIR=Path(os.getenv("ARTIFACT_DIR","artifacts"))
WINDOW=24
SEED=42

def catalog():
    if not DATA_DIR: raise SystemExit("Set CARE_DATA_DIR in .env or environment.")
    df=discover_events(DATA_DIR)
    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/event_catalog.csv",index=False)
    print(f"Found {len(df)} events across {df.farm.nunique()} farm folders.")
    print(df[["farm","event_id","event_label","asset_id","event_description"]].to_string(index=False))
    print("Saved results/event_catalog.csv")

def train(farm=None,event_id=None,epochs=25):
    if not DATA_DIR: raise SystemExit("Set CARE_DATA_DIR in .env or environment.")
    torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)
    catalog_df=discover_events(DATA_DIR)
    chosen=catalog_df
    if farm: chosen=chosen[chosen.farm.astype(str)==str(farm)]
    if event_id: chosen=chosen[chosen.event_id.astype(str)==str(event_id)]
    if chosen.empty: raise SystemExit("No event matched the supplied --farm / --event-id.")
    row=chosen.iloc[0]
    raw,frame,meta,sensor_meta=load_event(row)
    splits=prepare_splits(frame,meta,window=WINDOW)
    graphs,_=build_graphs(splits["train"],sensor_meta)
    # Fusion graph: union of available graph evidence, normalized later by model.
    adj=np.maximum.reduce(list(graphs.values())).astype(np.float32)
    Xtr,ytr=make_windows(splits["train"],WINDOW)
    Xv,yv=make_windows(splits["val"],WINDOW)
    Xt,yt=make_windows(splits["test"],WINDOW)
    if len(Xtr)==0 or len(Xv)==0 or len(Xt)==0: raise ValueError("Insufficient windows in one split; choose a longer event.")
    device="cuda" if torch.cuda.is_available() else "cpu"
    model=GraphTemporalForecaster(
        len(frame.columns), explicit_graphs=list(graphs.values()), window=WINDOW
    ).to(device)
    A=torch.tensor(adj,dtype=torch.float32,device=device)
    opt=torch.optim.Adam(model.parameters(),lr=1e-3,weight_decay=1e-5)
    loss_fn=torch.nn.L1Loss()
    best=float("inf"); patience=5; wait=0; best_state=None
    def batch_predict(X,batch=128):
        model.eval(); out=[]
        with torch.no_grad():
            for i in range(0,len(X),batch):
                xb=torch.tensor(X[i:i+batch],dtype=torch.float32,device=device)
                out.append(model(xb,A).cpu().numpy())
        return np.concatenate(out)
    for epoch in range(1,epochs+1):
        model.train()
        order=np.random.permutation(len(Xtr))
        for start in range(0,len(order),64):
            ids=order[start:start+64]
            xb=torch.tensor(Xtr[ids],dtype=torch.float32,device=device)
            yb=torch.tensor(ytr[ids],dtype=torch.float32,device=device)
            opt.zero_grad(); loss=loss_fn(model(xb,A),yb); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            opt.step()
        val_pred=batch_predict(Xv)
        val_loss=float(np.mean(np.abs(val_pred-yv)))
        print(f"epoch {epoch:02d}/{epochs} val_MAE={val_loss:.5f}")
        if val_loss<best:
            best=val_loss; wait=0; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            wait+=1
            if wait>=patience: break
    model.load_state_dict(best_state); model.to(device)
    test_pred=batch_predict(Xt)
    metrics=forecast_metrics(yt,test_pred)
    train_pred=batch_predict(Xtr)
    mu,sigma=residual_stats(train_pred-ytr)
    ARTIFACT_DIR.mkdir(parents=True,exist_ok=True)
    slug=f"{str(row.farm).replace(' ','_')}_{str(row.event_id).replace('/','_')}"
    ckpt=ARTIFACT_DIR/f"{slug}.pt"
    torch.save({"state_dict":model.state_dict(),"n_nodes":len(frame.columns),"window":WINDOW,"adjacency":adj,
                "explicit_graphs":{k:v for k,v in graphs.items()},
                "sensor_names":list(frame.columns),"sensor_meta":sensor_meta,"scale":splits["scale"],
                "mu":mu,"sigma":sigma,"test_metrics":metrics,"farm":str(row.farm),"event_id":str(row.event_id),
                "event_description":str(row.event_description),"event_csv":str(row.event_csv)},ckpt)
    print("Saved",ckpt)
    print("Held-out test metrics:",json.dumps(metrics,indent=2))
    Path("results").mkdir(exist_ok=True)
    with open(Path("results")/f"{slug}_metrics.json","w",encoding="utf-8") as f: json.dump(metrics,f,indent=2)

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest="command",required=True)
    sub.add_parser("catalog")
    t=sub.add_parser("train"); t.add_argument("--farm"); t.add_argument("--event-id"); t.add_argument("--epochs",type=int,default=25)
    args=p.parse_args()
    if args.command=="catalog": catalog()
    elif args.command=="train": train(args.farm,args.event_id,args.epochs)
if __name__=="__main__": main()
