import numpy as np

def gradient_saliency(model, x_window, adjacency, device="cpu"):
    import torch
    model.eval()
    x=torch.tensor(x_window[None],dtype=torch.float32,device=device,requires_grad=True)
    pred=model(x,adjacency)
    # Explain the largest predicted residual contribution; actual value is supplied by caller.
    target=pred.sum()
    model.zero_grad(set_to_none=True)
    target.backward()
    return np.abs(x.grad.detach().cpu().numpy()[0])

def shap_explain(model, x_window, adjacency, observed, mu, sigma, device="cpu", nsamples=80):
    """Optional Kernel SHAP on flattened lagged values; returns feature attribution vector."""
    try:
        import shap
        import torch
    except ImportError as exc:
        raise RuntimeError("SHAP is not installed. Run: pip install shap") from exc
    ntime,nodes=x_window.shape
    flat=x_window.reshape(1,-1)
    background=np.zeros((1,flat.shape[1]),dtype=np.float32)
    def score_fn(flat_batch):
        batch=[]
        for row in flat_batch:
            x=row.reshape(ntime,nodes)
            with torch.no_grad():
                pred=model(torch.tensor(x[None],dtype=torch.float32,device=device),adjacency).cpu().numpy()[0]
            # Scalar mean anomaly degree across nodes for the selected observed timestep.
            z=(pred-observed-mu)/(sigma+1e-8)
            batch.append([float(np.mean(1-np.exp(-.5*z*z)))])
        return np.asarray(batch)
    explainer=shap.KernelExplainer(score_fn,background)
    vals=explainer.shap_values(flat,nsamples=nsamples)
    vals=np.asarray(vals)
    vals=vals.reshape(-1)
    return vals.reshape(ntime,nodes)
