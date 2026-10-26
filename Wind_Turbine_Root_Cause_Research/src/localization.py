import numpy as np
import pandas as pd
import networkx as nx

def residual_stats(residuals):
    mu=np.mean(residuals,axis=0)
    sigma=np.std(residuals,axis=0)
    sigma=np.where(sigma<1e-6,1e-6,sigma)
    return mu,sigma

def anomaly_degree(y,pred,mu,sigma):
    z=(pred-y-mu)/(sigma+1e-8)
    return 1-np.exp(-0.5*np.square(z))

def earliest_onset(scores,names,threshold=.5):
    rows=[]
    for i,name in enumerate(names):
        hit=np.flatnonzero(scores[:,i]>=threshold)
        rows.append({"sensor":name,"onset":int(hit[0]) if len(hit) else None})
    return sorted(rows,key=lambda r: r["onset"] if r["onset"] is not None else 10**12)

def propagation_lags(scores,names,max_lag=30,threshold=.3):
    onset={r["sensor"]:r["onset"] for r in earliest_onset(scores,names,threshold)}
    rows=[]
    for i,src in enumerate(names):
        for j,dst in enumerate(names):
            if i==j: continue
            a=scores[:,i]; b=scores[:,j]
            best_lag,best=-1,-1
            for lag in range(max_lag+1):
                aa,bb=(a,b) if lag==0 else (a[:-lag],b[lag:])
                if len(aa)<8 or np.std(aa)<1e-8 or np.std(bb)<1e-8: continue
                corr=np.corrcoef(aa,bb)[0,1]
                if np.isfinite(corr) and corr>best: best,best_lag=corr,lag
            if best>=0.15:
                rows.append({"source":src,"target":dst,"lag":best_lag,"score":float(best),
                             "onset_delta":None if onset[src] is None or onset[dst] is None else onset[dst]-onset[src]})
    return pd.DataFrame(rows).sort_values("score",ascending=False) if rows else pd.DataFrame(columns=["source","target","lag","score","onset_delta"])

def rank_candidates(scores,names,graph,uncertainty=None,physical=None):
    scores=np.asarray(scores)
    n=len(names)
    peak=np.max(scores,axis=0)
    mean=np.mean(scores,axis=0)
    onset=earliest_onset(scores,names)
    onset_map={r["sensor"]:r["onset"] for r in onset}
    # Outgoing connectivity is supporting evidence only; do not interpret as causal truth.
    out_degree=np.asarray(graph).sum(axis=1)
    out_norm=out_degree/(out_degree.max()+1e-8)
    unc=np.zeros(n) if uncertainty is None else np.asarray(uncertainty).mean(axis=0)
    unc_norm=unc/(unc.max()+1e-8)
    physical=np.zeros(n) if physical is None else np.asarray(physical)
    rows=[]
    for i,name in enumerate(names):
        t=onset_map[name]
        early=0 if t is None else 1-(t/max(1,len(scores)-1))
        # Transparent heuristic composite; weights are configurable in the research protocol.
        score=.45*peak[i]+.20*mean[i]+.20*early+.10*out_norm[i]+.05*physical[i]-.05*unc_norm[i]
        rows.append({"sensor":name,"evidence_score":float(score),"peak_anomaly":float(peak[i]),
                     "mean_anomaly":float(mean[i]),"onset":t,"out_degree":int(out_degree[i]),
                     "uncertainty_penalty":float(unc_norm[i]),"physical_support":float(physical[i])})
    return pd.DataFrame(rows).sort_values("evidence_score",ascending=False).reset_index(drop=True)

def counterfactual_attenuation(model,predict_fn,x_window,observed,adjacency,base_scores,mu,sigma,names,device="cpu",attenuation=0.0):
    """Attenuate one candidate's latest input value; compare downstream anomaly-score changes.
    This is a model-input counterfactual, not a physical causal intervention.
    """
    import torch
    original=x_window.copy()
    rows=[]
    with torch.no_grad():
        base_pred=predict_fn(torch.tensor(original[None],dtype=torch.float32,device=device),adjacency).cpu().numpy()[0]
        base=anomaly_degree(observed,base_pred,mu,sigma)
        for i,name in enumerate(names):
            changed=original.copy()
            changed[-1,i]=changed[-1,i]*(1-attenuation)
            pred=predict_fn(torch.tensor(changed[None],dtype=torch.float32,device=device),adjacency).cpu().numpy()[0]
            cf=anomaly_degree(observed,pred,mu,sigma)
            delta=base-cf
            downstream=np.flatnonzero(np.asarray(adjacency)[i]>0)
            downstream_delta=float(delta[downstream].mean()) if len(downstream) else 0.0
            rows.append({"candidate":name,"total_anomaly_reduction":float(delta.sum()),
                         "downstream_anomaly_reduction":downstream_delta,
                         "changed_sensor":float(delta[i]),"interpretation":"model-based attenuation only"})
    return pd.DataFrame(rows).sort_values("downstream_anomaly_reduction",ascending=False)
