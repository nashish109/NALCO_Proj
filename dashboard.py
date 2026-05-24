import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(page_title="Machine Breakdown Predictor", layout="wide")
st.title("Machine Breakdown Prediction Dashboard")

results_path = Path("results.json")
predictions_path = Path("machine_failure_predictions.csv")

if not results_path.exists():
    st.error("Run python app.py first.")
    st.stop()

data = json.loads(results_path.read_text(encoding="utf-8"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("Current Status", data["machine_status"])
col2.metric("Remaining Life", f'{data["remaining_life"]} cycles')
col3.metric("Failure Risk", f'{data["failure_risk"]}%')
col4.metric("Model", data["model_name"])

st.info(f'Breakdown from latest row: {data["approx_failure_time"]}')
st.warning(f'Early warning row: {data["early_warning_row"]} | Predicted breakdown row: {data["predicted_failure_row"]}')
st.caption(f'Main sensor factor: {data["main_failure_factor"]}')

left, right = st.columns(2)
with left:
    st.subheader("Actual vs Predicted")
    st.image("results/actual_vs_predicted.png", use_container_width=True)
with right:
    st.subheader("Feature Importance")
    st.image("results/feature_importance.png", use_container_width=True)

shap_path = Path("results/shap_summary.png")
if shap_path.exists():
    st.subheader("SHAP Explainability")
    st.image(str(shap_path), use_container_width=True)

if predictions_path.exists():
    st.subheader("Prediction Table")
    df = pd.read_csv(predictions_path)
    st.dataframe(df.tail(150), use_container_width=True, hide_index=True)
