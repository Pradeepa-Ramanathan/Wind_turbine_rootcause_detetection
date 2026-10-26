import numpy as np
import pandas as pd

def prepare_splits(frame, meta, window=48, train_ratio=0.8, val_ratio=0.2):
    """Fit scale on chronological training data only; test tail is never used for fitting."""
    values = frame.to_numpy(dtype=np.float32)
    if len(values) < window * 3 + 3:
        raise ValueError(f"Need at least {window*3+3} rows; event only has {len(values)}.")
    if "train_test" in meta.columns:
        tokens = meta["train_test"].astype(str).str.lower().str.strip()
        train_mask = tokens.isin({"train", "training", "0"})
        test_mask = ~train_mask
        if train_mask.sum() >= window * 2 + 2 and test_mask.sum() >= window + 1:
            train_full = values[train_mask.to_numpy()]
            test_raw = values[test_mask.to_numpy()]
        else:
            train_full, test_raw = _chronological_fallback(values, train_ratio)
    else:
        train_full, test_raw = _chronological_fallback(values, train_ratio)
    split = max(window + 1, int(len(train_full) * (1 - val_ratio)))
    split = min(split, len(train_full) - 1)
    fit_values = train_full[:split]
    val_raw = train_full[split:]
    scale = np.nanquantile(np.abs(fit_values), 0.99, axis=0)
    scale = np.where((~np.isfinite(scale)) | (scale < 1e-6), 1.0, scale).astype(np.float32)
    train = np.clip(fit_values / scale, -5, 5)
    val = np.clip(val_raw / scale, -5, 5)
    test = np.clip(test_raw / scale, -5, 5)
    # Use tail of train partition as validation; ensure enough windows.
    if len(val) <= window:
        val = train[max(0, len(train)-window-1):]
    return {"train": train, "val": val, "test": test, "scale": scale,
            "train_raw": train_full, "test_raw": test_raw}

def _chronological_fallback(values, train_ratio):
    cut = int(len(values) * train_ratio)
    cut = max(2, min(cut, len(values)-2))
    return values[:cut], values[cut:]
