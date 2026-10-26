import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import torch
from dotenv import load_dotenv

from src.care_data import discover_events, load_event
from src.preprocessing import prepare_splits
from src.graphs import event_window_graphs
from src.model import GraphTemporalForecaster, make_windows
from src.localization import (
    anomaly_degree,
    rank_candidates,
    propagation_lags,
    counterfactual_attenuation,
)
from src.explain import shap_explain, gradient_saliency


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
load_dotenv()

DATA_DIR = os.getenv("CARE_DATA_DIR", "")
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", "artifacts"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

st.set_page_config(
    page_title="Wind Turbine Root-Cause Digital Twin",
    layout="wide",
)

st.title("Wind Turbine Root-Cause Localization")
st.caption("Digital twin dashboard")


# ---------------------------------------------------------
# Load dataset and trained model
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_catalog(path):
    return discover_events(path)


@st.cache_resource(show_spinner=False)
def load_checkpoint(path):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)

    explicit_graphs = list(ck.get("explicit_graphs", {}).values())

    model = GraphTemporalForecaster(
        ck["n_nodes"],
        explicit_graphs=explicit_graphs,
        window=ck.get("window", 24),
    ).to(DEVICE)

    model.load_state_dict(ck["state_dict"])
    model.eval()

    adjacency = torch.tensor(
        ck["adjacency"],
        dtype=torch.float32,
        device=DEVICE,
    )

    return ck, model, adjacency


if not DATA_DIR or not Path(DATA_DIR).exists():
    st.warning(
        "CARE dataset not configured. Set CARE_DATA_DIR "
        "to the extracted dataset root."
    )
    st.code(
        "python -m src.cli catalog\n"
        "python -m src.cli train\n"
        "streamlit run app.py"
    )
    st.stop()


try:
    catalog = get_catalog(DATA_DIR)
except Exception as exc:
    st.error(f"Could not load CARE catalog: {exc}")
    st.stop()


# ---------------------------------------------------------
# Event selection
# ---------------------------------------------------------
with st.sidebar:
    st.header("Event selection")

    farm = st.selectbox(
        "Wind farm",
        sorted(catalog.farm.unique()),
    )

    farm_events = catalog[catalog.farm == farm]

    event_id = st.selectbox(
        "Event ID",
        farm_events.event_id.astype(str).tolist(),
    )

    selected = farm_events[
        farm_events.event_id.astype(str) == event_id
    ].iloc[0]

    st.write("**Label:**", selected.event_label)
    st.write("**Asset:**", selected.asset_id)
    st.write("**Description:**", selected.event_description)


event_label = str(selected.get("event_label", "Unknown"))
label_lower = event_label.strip().lower()

is_normal_event = (
    "normal" in label_lower
    and not any(
        word in label_lower
        for word in ("abnormal", "anomaly", "fault", "failure")
    )
)


# ---------------------------------------------------------
# Reuse the existing trained checkpoint
# ---------------------------------------------------------
artifact_files = sorted(
    glob.glob(str(ARTIFACT_DIR / "*.pt"))
)

if not artifact_files:
    st.error(
        f"No trained checkpoint found in {ARTIFACT_DIR}. "
        "Train the model once with `python -m src.cli train`."
    )
    st.stop()


artifact = artifact_files[0]
ck, model, A = load_checkpoint(artifact)

st.sidebar.caption(
    f"Using trained model: {Path(artifact).name}"
)


try:
    raw, frame, meta, sensor_meta = load_event(
        selected,
        sensor_names=ck["sensor_names"],
    )
except Exception as exc:
    st.error(f"Could not load selected event: {exc}")
    st.stop()


# ---------------------------------------------------------
# Event summary
# ---------------------------------------------------------
st.subheader("Event Summary")

st.write(f"**Original event label:** {event_label}")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Rows", f"{len(frame):,}")
m2.metric("Sensor channels", len(frame.columns))
m3.metric("Farm", str(farm))
m4.metric("Event", str(event_id))


tabs = st.tabs(
    [
        "Sensor explorer",
        "Anomaly & root cause",
        "Graph & propagation",
        "Counterfactual checking",
        "SHAP explanation",
    ]
)


# ---------------------------------------------------------
# Normal event handling
# ---------------------------------------------------------
if is_normal_event:
    for tab in tabs:
        with tab:
            st.write("This is a normal event.")
    st.stop()


# ---------------------------------------------------------
# Prepare model inputs and calculate sensor anomaly degrees
# ---------------------------------------------------------
names = list(ck["sensor_names"])

splits = prepare_splits(
    frame,
    meta,
    window=ck["window"],
)

test = splits["test"]
X, y = make_windows(test, ck["window"])

if len(X) == 0:
    for tab in tabs:
        with tab:
            st.warning(
                "Not enough test rows to create model windows "
                "for this event."
            )
    st.stop()


predictions = []

with torch.no_grad():
    for i in range(0, len(X), 128):
        xb = torch.tensor(
            X[i:i + 128],
            dtype=torch.float32,
            device=DEVICE,
        )

        predictions.append(
            model(xb, A).cpu().numpy()
        )


pred = np.concatenate(predictions, axis=0)

mu = np.asarray(ck["mu"])
sigma = np.asarray(ck["sigma"])

scores = anomaly_degree(
    y,
    pred,
    mu,
    sigma,
)

uncertainty = np.zeros_like(scores)

ranked = rank_candidates(
    scores,
    names,
    ck["adjacency"],
    uncertainty,
)


# ---------------------------------------------------------
# Tab 1: Sensor explorer
# ---------------------------------------------------------
with tabs[0]:
    st.subheader("Sensor time series")

    plot_df = (
        frame.reset_index()
        .melt(
            id_vars="index",
            var_name="sensor",
            value_name="value",
        )
    )

    fig = px.line(
        plot_df,
        x="index",
        y="value",
        color="sensor",
        title="Observed values across all model sensors",
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Sensor metadata")

    metadata_df = pd.DataFrame(
        [
            {"sensor": sensor, **details}
            for sensor, details in sensor_meta.items()
        ]
    )

    st.dataframe(
        metadata_df,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Tab 2: Anomaly and root-cause analysis
# ---------------------------------------------------------
with tabs[1]:
    st.subheader("Anomaly & root cause")

    anomaly_df = pd.DataFrame(
        scores,
        columns=names,
    )

    anomaly_df["timestep"] = np.arange(
        len(anomaly_df)
    )

    fig = px.line(
        anomaly_df,
        x="timestep",
        y=names,
        title="Anomaly degree across all 20 sensors",
        labels={
            "timestep": "Time step",
            "value": "Anomaly degree",
            "variable": "Sensor",
        },
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Ranked root-cause candidates")

    st.dataframe(
        ranked,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Tab 3: Graph and propagation
# ---------------------------------------------------------
with tabs[2]:
    st.subheader("20-sensor graph and propagation")

    adjacency = np.asarray(ck["adjacency"])

    graph = nx.from_numpy_array(
        adjacency,
        create_using=nx.DiGraph,
    )

    graph = nx.relabel_nodes(
        graph,
        {
            i: names[i]
            for i in range(
                min(len(names), adjacency.shape[0])
            )
        },
    )

    pos = nx.spring_layout(
        graph,
        seed=42,
        k=1.1,
    )

    edge_x = []
    edge_y = []

    for source, target in graph.edges():
        edge_x.extend(
            [
                pos[source][0],
                pos[target][0],
                None,
            ]
        )

        edge_y.extend(
            [
                pos[source][1],
                pos[target][1],
                None,
            ]
        )

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=edge_x,
            y=edge_y,
            mode="lines",
            line=dict(width=1.3),
            hoverinfo="none",
            name="Connections",
        )
    )

    for source, target in graph.edges():
        x0, y0 = pos[source]
        x1, y1 = pos[target]

        fig.add_annotation(
            x=x1,
            y=y1,
            ax=x0,
            ay=y0,
            xref="x",
            yref="y",
            axref="x",
            ayref="y",
            showarrow=True,
            arrowhead=3,
            arrowsize=1,
            arrowwidth=1.2,
        )

    if "sensor" in ranked.columns and "evidence_score" in ranked.columns:
        scores_by_sensor = (
            ranked.set_index("sensor")["evidence_score"]
            .reindex(names)
            .fillna(0)
        )
    else:
        scores_by_sensor = pd.Series(
            0.0,
            index=names,
        )

    max_evidence = (
        float(scores_by_sensor.max())
        if len(scores_by_sensor)
        else 0.0
    )

    marker_sizes = [
        14 + 24 * float(
            scores_by_sensor.get(sensor, 0)
            / (max_evidence + 1e-8)
        )
        for sensor in graph.nodes()
    ]

    fig.add_trace(
        go.Scatter(
            x=[pos[node][0] for node in graph.nodes()],
            y=[pos[node][1] for node in graph.nodes()],
            mode="markers+text",
            text=list(graph.nodes()),
            textposition="top center",
            marker=dict(size=marker_sizes),
            hovertext=[
                (
                    f"{sensor}: evidence="
                    f"{scores_by_sensor.get(sensor, 0):.3f}"
                )
                for sensor in graph.nodes()
            ],
            hoverinfo="text",
            name="Sensors",
        )
    )

    fig.update_layout(
        title="Directed sensor connections (arrows show edge direction)",
        showlegend=False,
        height=650,
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.subheader("Propagation lag candidates")

    lags = propagation_lags(
        scores,
        names,
    )

    st.dataframe(
        lags,
        use_container_width=True,
    )

    st.subheader("Event-based graph snapshots")

    windows = event_window_graphs(test)

    if windows:
        snapshots_df = pd.DataFrame(
            [
                {
                    "start": item["start"],
                    "end": item["end"],
                    "edges": int(
                        item["adjacency"].sum()
                    ),
                }
                for item in windows
            ]
        )

        st.dataframe(
            snapshots_df,
            use_container_width=True,
        )
    else:
        st.info(
            "No event-based graph snapshots were generated "
            "for this event."
        )


# ---------------------------------------------------------
# Tab 4: Counterfactual checking
# ---------------------------------------------------------
with tabs[3]:
    st.subheader("Counterfactual checking")

    try:
        cf = counterfactual_attenuation(
            model,
            lambda xb, adj: model(xb, adj),
            X[-1],
            y[-1],
            A.cpu().numpy(),
            scores,
            mu,
            sigma,
            names,
            DEVICE,
            0.5,
        )

        if cf is None or cf.empty:
            st.info(
                "Counterfactual analysis did not return "
                "any candidate results."
            )

        else:
            cf_display = cf.copy()

            reduction_col = "downstream_anomaly_reduction"

            if reduction_col in cf_display.columns:
                numeric_reduction = pd.to_numeric(
                    cf_display[reduction_col],
                    errors="coerce",
                )

                # Preserve the original value for calculations.
                cf_display["_reduction_numeric"] = (
                    numeric_reduction
                )

                # Rank the candidates by the returned reduction.
                cf_display = cf_display.sort_values(
                    "_reduction_numeric",
                    ascending=False,
                    na_position="last",
                ).reset_index(drop=True)

                valid_values = (
                    cf_display["_reduction_numeric"]
                    .dropna()
                )

                n_valid = len(valid_values)

                def interpret_reduction(position, total):
                    if total <= 1:
                        return (
                            "Only candidate available; "
                            "compare with other evidence."
                        )

                    percentile = position / (total - 1)

                    if percentile <= 0.25:
                        return (
                            "Higher relative impact: "
                            "this candidate produced one of "
                            "the largest reductions in this comparison."
                        )

                    if percentile <= 0.60:
                        return (
                            "Moderate relative impact: "
                            "the reduction is between the stronger "
                            "and weaker candidates."
                        )

                    return (
                        "Lower relative impact: "
                        "this candidate produced a smaller reduction "
                        "than most other candidates."
                    )

                interpretations = []
                positions = []

                for idx, value in enumerate(
                    cf_display["_reduction_numeric"]
                ):
                    if pd.isna(value):
                        interpretations.append(
                            "Impact could not be evaluated "
                            "from the returned value."
                        )
                        positions.append("Not ranked")
                    else:
                        rank_position = int(
                            (
                                cf_display["_reduction_numeric"]
                                .dropna()
                                .gt(value)
                            ).sum()
                        )

                        positions.append(
                            f"{rank_position + 1} of {n_valid}"
                        )

                        interpretations.append(
                            interpret_reduction(
                                rank_position,
                                n_valid,
                            )
                        )

                cf_display["Relative ranking"] = positions
                cf_display["Interpretation"] = interpretations

                # Add a percentage-formatted column for readability.
                # This is a display conversion of the returned value,
                # not a new model prediction.
                cf_display["Reduction value"] = (
                    cf_display["_reduction_numeric"]
                    .map(
                        lambda value: (
                            f"{value:.6f}"
                            if pd.notna(value)
                            else "Unavailable"
                        )
                    )
                )

                display_columns = [
                    col
                    for col in cf_display.columns
                    if col != "_reduction_numeric"
                    and col != reduction_col
                ]

                # Place the readable reduction column next to candidate.
                if "candidate" in display_columns:
                    display_columns.remove("Reduction value")
                    candidate_index = display_columns.index(
                        "candidate"
                    )
                    display_columns.insert(
                        candidate_index + 1,
                        "Reduction value",
                    )

                st.dataframe(
                    cf_display[display_columns],
                    use_container_width=True,
                    hide_index=True,
                )

                # The chart emphasizes relative differences between
                # candidate sensors instead of relying only on tiny
                # raw numerical values.
                chart_df = cf_display[
                    ["candidate", "_reduction_numeric"]
                ].dropna().copy()

                if not chart_df.empty:
                    chart_df = chart_df.rename(
                        columns={
                            "_reduction_numeric":
                                "Downstream anomaly reduction"
                        }
                    )

                    fig = px.bar(
                        chart_df,
                        x="candidate",
                        y="Downstream anomaly reduction",
                        title=(
                            "Counterfactual impact by candidate "
                            "(larger values indicate greater reduction)"
                        ),
                        labels={
                            "candidate": "Candidate sensor",
                            "Downstream anomaly reduction":
                                "Returned reduction value",
                        },
                    )

                    fig.update_layout(
                        xaxis_tickangle=-45,
                    )

                    st.plotly_chart(
                        fig,
                        use_container_width=True,
                    )

                st.caption(
                    "Interpretation is relative to the candidates "
                    "evaluated for this event. A larger reduction "
                    "indicates greater model sensitivity to that "
                    "candidate under the tested counterfactual; it "
                    "does not by itself prove a physical cause."
                )

            else:
                st.dataframe(
                    cf_display,
                    use_container_width=True,
                    hide_index=True,
                )

                st.info(
                    "The counterfactual output does not contain "
                    "the expected downstream reduction column, "
                    "so the returned results are shown as provided."
                )

    except Exception as exc:
        st.warning(
            f"Counterfactual analysis could not be computed: {exc}"
        )


# ---------------------------------------------------------
# Tab 5: SHAP explanation
# ---------------------------------------------------------
with tabs[4]:
    st.subheader("SHAP explanation")

    try:
        with st.spinner("Computing SHAP attribution..."):
            attr = shap_explain(
                model,
                X[-1],
                A,
                y[-1],
                mu,
                sigma,
                DEVICE,
                nsamples=60,
            )

        by_sensor = np.abs(attr).mean(axis=0)

        shap_df = pd.DataFrame(
            {
                "sensor": names,
                "mean_abs_shap": by_sensor,
            }
        ).sort_values(
            "mean_abs_shap",
            ascending=True,
        )

        st.plotly_chart(
            px.bar(
                shap_df,
                x="mean_abs_shap",
                y="sensor",
                orientation="h",
                title="SHAP attribution by sensor",
                labels={
                    "mean_abs_shap": "Mean absolute SHAP attribution",
                    "sensor": "Sensor",
                },
            ),
            use_container_width=True,
        )

        heat = pd.DataFrame(
            attr.T,
            index=names,
            columns=[
                f"t-{ck['window'] - i}"
                for i in range(ck["window"])
            ],
        )

        st.plotly_chart(
            px.imshow(
                heat,
                aspect="auto",
                title="SHAP contribution by sensor and time step",
                labels={
                    "x": "Time step",
                    "y": "Sensor",
                    "color": "SHAP attribution",
                },
            ),
            use_container_width=True,
        )

    except Exception as exc:
        st.warning(f"SHAP could not run: {exc}")

        st.subheader("Gradient saliency")

        try:
            sal = gradient_saliency(
                model,
                X[-1],
                A,
                DEVICE,
            )

            st.plotly_chart(
                px.imshow(
                    sal.T,
                    aspect="auto",
                    labels={
                        "x": "Time step",
                        "y": "Sensor",
                        "color": "Saliency",
                    },
                    y=names,
                    title="Gradient saliency by sensor and time step",
                ),
                use_container_width=True,
            )

        except Exception as saliency_exc:
            st.error(
                f"Gradient saliency could not be computed: "
                f"{saliency_exc}"
            )
