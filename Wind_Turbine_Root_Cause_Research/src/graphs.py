import numpy as np
from sklearn.feature_selection import mutual_info_regression
from .care_data import category_for

def _empty(n): return np.zeros((n, n), dtype=np.float32)

def mechanical_graph(sensor_meta):
    n=len(sensor_meta); A=_empty(n)
    winds=[i for i,m in enumerate(sensor_meta.values()) if m["category"]=="wind"]
    speeds=[i for i,m in enumerate(sensor_meta.values()) if m["category"]=="speed"]
    powers=[i for i,m in enumerate(sensor_meta.values()) if m["category"]=="power"]
    for w in winds:
        for s in speeds: A[w,s]=1
    for k in range(len(speeds)-1):
        A[speeds[k],speeds[k+1]]=1
    for s in speeds:
        for p in powers: A[s,p]=1
    return A

def temperature_graph(sensor_meta):
    n=len(sensor_meta); A=_empty(n)
    items=list(sensor_meta.items())
    powers=[i for i,(_,m) in enumerate(items) if m["category"]=="power"]
    temps=[i for i,(_,m) in enumerate(items) if m["category"]=="temperature"]
    for p in powers:
        for t in temps: A[p,t]=1
    for i in temps:
        for j in temps:
            if i!=j and any(k in items[i][1]["description"].lower() and k in items[j][1]["description"].lower()
                            for k in ["gearbox","generator","converter"]):
                A[i,j]=1
    return A

def correlation_graph(x, threshold=.7):
    c=np.corrcoef(x.T)
    c=np.nan_to_num(c)
    A=(np.abs(c)>=threshold).astype(np.float32); np.fill_diagonal(A,0)
    return A,c

def mutual_information_graph(x, top_k=3, max_samples=2500, seed=42):
    n=x.shape[1]; rng=np.random.default_rng(seed)
    sub=x[rng.choice(len(x),min(len(x),max_samples),replace=False)]
    mi=np.zeros((n,n),dtype=np.float32)
    for j in range(n):
        mi[:,j]=mutual_info_regression(sub,sub[:,j],random_state=seed)
    mi=(mi+mi.T)/2; np.fill_diagonal(mi,0)
    A=_empty(n)
    for i in range(n):
        for j in np.argsort(mi[i])[::-1]:
            if i!=j and mi[i,j]>0:
                A[i,j]=1
                if A[i].sum()>=top_k: break
    A=np.maximum(A,A.T); np.fill_diagonal(A,0)
    return A,mi

def _granger_improvement(x, y, lag=3):
    T=len(y)
    if T <= 2*lag+5: return 0.
    target=y[lag:]
    yr=np.column_stack([y[lag-k:T-k] for k in range(1,lag+1)])
    xf=np.column_stack([yr]+[x[lag-k:T-k] for k in range(1,lag+1)])
    def rss(a):
        a=np.column_stack([np.ones(len(a)),a])
        b=np.linalg.lstsq(a,target,rcond=None)[0]
        return np.square(target-a@b).sum()
    rr,rf=rss(yr),rss(xf)
    return float(np.clip((rr-rf)/(rr+1e-9),0,1))

def granger_graph(x, lag=3, top_k=2, max_rows=2500):
    # Computational cap for interactive use; use the full notebook for large causal-graph runs.
    x=x[:max_rows]; n=x.shape[1]; scores=np.zeros((n,n),np.float32)
    for i in range(n):
        for j in range(n):
            if i!=j: scores[i,j]=_granger_improvement(x[:,i],x[:,j],lag)
    A=_empty(n)
    for effect in range(n):
        for cause in np.argsort(scores[:,effect])[::-1]:
            if cause!=effect and scores[cause,effect]>1e-4:
                A[cause,effect]=1
                if A[:,effect].sum()>=top_k: break
    return A,scores

def build_graphs(train, sensor_meta):
    corr,corr_raw=correlation_graph(train)
    mi,mi_raw=mutual_information_graph(train)
    causal,causal_raw=granger_graph(train)
    return {
        "Mechanical (heuristic)": mechanical_graph(sensor_meta),
        "Temperature (heuristic)": temperature_graph(sensor_meta),
        "Correlation": corr,
        "Mutual information": mi,
        "Granger substitute (not CORL)": causal
    }, {"correlation":corr_raw,"mutual_information":mi_raw,"granger":causal_raw}

def event_window_graphs(x, window=96, step=48):
    """Exploratory correlation graphs for before/during/after windows, not causal proof."""
    out=[]
    for start in range(0,max(1,len(x)-window+1),step):
        seg=x[start:start+window]
        if len(seg)<max(20,x.shape[1]+2): continue
        A,_=correlation_graph(seg,.7)
        out.append({"start":start,"end":start+len(seg),"adjacency":A})
    return out
