import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, precision_recall_fscore_support, average_precision_score

def forecast_metrics(y_true,y_pred):
    return {"mae":float(mean_absolute_error(y_true,y_pred)),
            "rmse":float(np.sqrt(mean_squared_error(y_true,y_pred)))}

def anomaly_metrics(y_true,pred_score,threshold=.5):
    y_true=np.asarray(y_true).astype(int)
    y_pred=(np.asarray(pred_score)>=threshold).astype(int)
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average="binary",zero_division=0)
    try: ap=average_precision_score(y_true,pred_score)
    except ValueError: ap=float("nan")
    return {"precision":float(p),"recall":float(r),"f1":float(f),"pr_auc":float(ap)}

def ranking_stability(rankings):
    """Rankings: list of sensor-name lists ordered best to worst."""
    if len(rankings)<2: return {"mean_top1_agreement":float("nan"),"mean_jaccard_top3":float("nan")}
    top1=np.mean([r[0]==rankings[0][0] for r in rankings[1:] if r and rankings[0]])
    sets=[set(r[:3]) for r in rankings]
    vals=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            vals.append(len(sets[i]&sets[j])/max(1,len(sets[i]|sets[j])))
    return {"mean_top1_agreement":float(top1),"mean_jaccard_top3":float(np.mean(vals))}
