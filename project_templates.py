DASHBOARD_CODE = '''import json
from pathlib import Path

import streamlit as st


st.set_page_config(page_title="Machine Breakdown Predictor", layout="wide")
st.title("Machine Breakdown Status")

results_path = Path("results.json")
graph_path = Path("results/actual_vs_predicted.png")

if not results_path.exists():
    st.error("Run python app.py first.")
    st.stop()

data = json.loads(results_path.read_text(encoding="utf-8"))

for warning in data.get("warnings", []):
    st.warning(warning)

status_col, time_col = st.columns(2)
status_col.metric("Machine Status", data["machine_status"])
time_col.metric("Breakdown Time", data["approx_failure_time"])

rul_col, model_col = st.columns(2)
rul_col.metric(
    "Remaining Life",
    f"{data['remaining_life']} {data.get('remaining_life_unit', 'cycles')}",
)
model_col.metric("Model", data.get("model_name", "ridge_regression"))

st.subheader("Actual vs Predicted")
if graph_path.exists():
    st.image(str(graph_path), use_container_width=True)
else:
    st.warning("Run python app.py to generate the actual vs predicted graph.")
'''
