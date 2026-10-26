# Wind Turbine Root-Cause Localization

A VS Code research project for CARE-to-Compare SCADA event analysis. The Streamlit dashboard retains the interactive event exploration, forecasting evaluation, anomaly scoring, graph visualization, counterfactual analysis, and explanation views. The forecasting model is based on the research implementation: learned implicit graph + attention-weighted explicit graph fusion + MTGNN-style temporal/graph convolutions.

## 1. Project layout

```text
Wind_Turbine_Root_Cause_Research/
├── app.py                       # Streamlit dashboard
├── .env.example                 # Environment-variable template
├── requirements.txt
├── src/
│   ├── care_data.py             # CARE event discovery and sensor metadata
│   ├── preprocessing.py         # Chronological splits and train-only scaling
│   ├── graphs.py                # Mechanical, temperature, correlation, MI, causal graphs
│   ├── model.py                 # Implicit graph learning, knowledge fusion, MTGNN
│   ├── localization.py          # Anomaly and root-cause evidence
│   ├── explain.py               # Gradient saliency and optional SHAP
│   ├── experiment.py            # Forecast/anomaly metrics and ranking stability
│   └── cli.py                   # Catalog and training commands
├── research/
│   └── Wind_Turbine_Root_Cause_Localization.ipynb
├── artifacts/                   # Generated checkpoints; ignored by Git
└── results/                     # Generated metrics/catalog; ignored by Git
```

The research notebook preserves the detailed experiment flow and baseline comparisons. The modular Python model is the implementation used by the CLI and dashboard; the dashboard does not need to execute the notebook.

## 2. Keep the dataset outside this repository

Extract CARE-to-Compare to any location you control. Do not copy it into this project. Copy `.env.example` to `.env` and set the path to the extracted dataset root (the folder containing the farm folders or `event_info.csv` files):

```dotenv
CARE_DATA_DIR="D:/datasets/CARE-to-Compare"
ARTIFACT_DIR="artifacts"
```

Paths in `.env` are local machine configuration and must not be committed. If your dataset is in a different location, only change `CARE_DATA_DIR`.

## 3. Install (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Edit `.env` before continuing. Python 3.10/3.11 is recommended. Install the VS Code Python and Jupyter extensions if you plan to run the research notebook.

## 4. Discover events and train the model

```powershell
python -m src.cli catalog
python -m src.cli train --epochs 25
```

To select an event explicitly:

```powershell
python -m src.cli train --farm "FARM_FOLDER_NAME" --event-id "EVENT_ID" --epochs 25
```

Use the farm and event IDs printed by `catalog`. Training uses chronological partitions and fits the scaler on training data only. A trained checkpoint is written under `ARTIFACT_DIR`; metrics and the event catalog are written under `results/`. Start with one event to validate the setup before running broader experiments.

## 5. Launch the dashboard

```powershell
streamlit run app.py
```

The dashboard discovers the configured dataset, lets you select an event, and enables model-dependent views when a matching trained checkpoint exists. Train first if the sidebar has no checkpoint for the selected event.

## 6. Research implementation

The proposed pipeline combines:
- Five explicit sensor graphs: mechanical, temperature-related, correlation, mutual information, and directed lag/causal evidence.
- A trainable implicit sensor graph learned from node embeddings.
- Knowledge fusion where attention weights determine the contribution of explicit graphs to the implicit graph.
- MTGNN-style dilated inception temporal convolutions and mix-hop graph propagation.
- Forecast residual anomaly degree, root-cause candidate ranking, propagation-delay evidence, uncertainty-aware analysis, and counterfactual input attenuation.

These outputs are **candidate evidence**, not a certified diagnosis. CARE event descriptions are generally component-level and do not provide sensor-level root-cause ground truth for every event. Do not claim sensor-level Top-k accuracy or superiority over baselines unless the relevant labels and repeated, leakage-safe experiments support it.

## 7. Reproducibility and review notes

- Keep raw datasets, `.env`, trained checkpoints, generated exports, caches, and notebook execution outputs out of source control.
- Review the event-level split and sensor preprocessing assumptions before reporting results.
- Record the event IDs, random seeds, configuration, and dependency versions for each experiment.
- The first run can take time because graph construction and neural model training are data-dependent.
- The dashboard's counterfactual attenuation is an input-level model probe, not proof of a physical causal intervention.
