from pathlib import Path
import pandas as pd

META_FILES = {"event_info.csv", "feature_description.csv"}

def read_care_csv(path):
    path = Path(path)
    last_error = None
    for sep in (",", ";"):
        try:
            df = pd.read_csv(path, sep=sep)
            if df.shape[1] > 1:
                df.columns = [str(c).strip() for c in df.columns]
                return df
        except Exception as exc:
            last_error = exc
    raise ValueError(f"Could not parse {path}: {last_error}")

def discover_events(data_dir):
    root = Path(data_dir).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"CARE data directory not found: {root}. Set CARE_DATA_DIR or edit .env.")
    info_paths = sorted(root.rglob("event_info.csv"))
    if not info_paths:
        raise FileNotFoundError(f"No event_info.csv found under {root}")
    records = []
    for info_path in info_paths:
        farm_dir = info_path.parent
        feature_path = farm_dir / "feature_description.csv"
        if not feature_path.exists():
            raise FileNotFoundError(f"Missing feature_description.csv beside {info_path}")
        events = read_care_csv(info_path)
        features = read_care_csv(feature_path)
        required = {"event_id", "event_label", "event_start", "event_end", "event_start_id", "event_end_id"}
        missing = required - set(events.columns)
        if missing:
            raise ValueError(f"{info_path} missing event columns: {sorted(missing)}")
        f_required = {"sensor_name", "description", "unit", "is_angle", "is_counter"}
        missing = f_required - set(features.columns)
        if missing:
            raise ValueError(f"{feature_path} missing sensor columns: {sorted(missing)}")
        csv_by_stem = {p.stem: p for p in farm_dir.rglob("*.csv") if p.name.lower() not in META_FILES}
        for _, row in events.iterrows():
            event_id = str(row["event_id"])
            csv_path = csv_by_stem.get(event_id)
            if csv_path is None:
                continue
            records.append({
                "farm": farm_dir.name, "event_id": event_id,
                "event_label": str(row.get("event_label", "")),
                "asset_id": str(row.get("asset_id", "")),
                "event_start": str(row.get("event_start", "")),
                "event_end": str(row.get("event_end", "")),
                "event_start_id": str(row.get("event_start_id", "")),
                "event_end_id": str(row.get("event_end_id", "")),
                "event_description": str(row.get("event_description", row.get("description", ""))),
                "event_csv": str(csv_path), "farm_dir": str(farm_dir),
                "feature_csv": str(feature_path)
            })
    if not records:
        raise ValueError("No event CSVs matched event_id stems. Check the CARE dataset folder layout.")
    return pd.DataFrame(records).sort_values(["farm", "event_id"]).reset_index(drop=True)

def load_event(event_row, max_sensors=20, sensor_names=None):
    event_row = dict(event_row)
    raw = read_care_csv(event_row["event_csv"])
    features = read_care_csv(event_row["feature_csv"])
    if "statistic_type" not in features:
        features["statistic_type"] = "avg"
    features["statistic_type"] = features["statistic_type"].astype(str).str.lower().str.strip()
    features["sensor_name"] = features["sensor_name"].astype(str).str.strip()
    features = features[features["statistic_type"].isin(["avg", "average"])].copy()
    meta_cols = [c for c in ["time_stamp", "id", "asset_id", "status_type_id", "train_test"] if c in raw.columns]
    available = [c for c in raw.columns if c not in set(meta_cols) | {"train_test"}]

    if sensor_names is not None:
        # Inference must use the exact sensor set and order expected by the trained model.
        missing = [name for name in sensor_names if name not in raw.columns]
        if missing:
            raise ValueError(
                "This event does not contain all sensors required by the trained checkpoint. "
                f"Missing {len(missing)} of {len(sensor_names)} sensors: {missing}"
            )
        columns = list(sensor_names)
        descriptions = {}
        feature_rows = features.set_index("sensor_name", drop=False)
        for col in columns:
            if col in feature_rows.index:
                row = feature_rows.loc[col]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                descriptions[col] = {"description": str(row.get("description", col)),
                                     "unit": str(row.get("unit", "")),
                                     "category": category_for(str(row.get("description", col)))}
            else:
                descriptions[col] = {"description": col, "unit": "", "category": category_for(col)}
    else:
        known = features[features["sensor_name"].isin(available)].copy()
        if known.empty:
            numeric = raw[available].apply(pd.to_numeric, errors="coerce")
            columns = [c for c in numeric.columns if numeric[c].notna().sum() > 0][:max_sensors]
            descriptions = {c: {"description": c, "unit": "", "category": category_for(c)} for c in columns}
        else:
            cols = known["sensor_name"].tolist()[:max_sensors]
            descriptions = {}
            for _, row in known.head(max_sensors).iterrows():
                col = row["sensor_name"]
                descriptions[col] = {"description": str(row.get("description", col)),
                                     "unit": str(row.get("unit", "")),
                                     "category": category_for(str(row.get("description", col)))}
            columns = [c for c in cols if c in raw.columns]
    if not columns:
        raise ValueError("No sensor columns could be matched. Inspect feature_description.csv and event CSV headers.")
    numeric = raw[columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.interpolate(limit_direction="both").ffill().bfill()
    keep = numeric.notna().all()
    numeric = numeric.loc[:, keep]
    if sensor_names is not None and list(numeric.columns) != list(sensor_names):
        raise ValueError("Required model sensor columns contain missing/non-numeric values after preprocessing.")
    descriptions = {c: descriptions.get(c, {"description": c, "unit": "", "category": category_for(c)}) for c in numeric.columns}
    meta = raw[meta_cols].copy() if meta_cols else pd.DataFrame(index=raw.index)
    return raw, numeric.astype("float32"), meta, descriptions

def category_for(text):
    s = str(text).lower()
    if "wind" in s: return "wind"
    if "power" in s: return "power"
    if "speed" in s or "rpm" in s or "rotor" in s or "generator" in s: return "speed"
    if "temp" in s or "temperature" in s: return "temperature"
    return "other"
