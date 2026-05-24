

import time
import json

import streamlit as st

from PIL import Image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(

    page_title="AI Predictive Maintenance",

    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title(
    "LIVE AI Predictive Maintenance Dashboard"
)

st.markdown(
    "Real-Time Machine Breakdown Forecasting"
)


# ============================================================
# AUTO REFRESH
# ============================================================

REFRESH_TIME = 5


# ============================================================
# LOAD RESULTS
# ============================================================

with open(
    "results.json",
    "r",
    encoding="utf-8"
) as f:

    data = json.load(f)


# ============================================================
# METRICS
# ============================================================

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Machine Status",
        data["machine_status"]
    )


with col2:

    st.metric(
        "Remaining Life",
        f'{data["remaining_life"]} cycles'
    )


with col3:

    st.metric(
        "Failure Risk",
        f'{data["failure_risk"]}%'
    )


# ============================================================
# FAILURE TIME
# ============================================================

st.info(
    f'Approx Failure Time: '
    f'{data["approx_failure_time"]}'
)


# ============================================================
# FAILURE FACTOR
# ============================================================

st.warning(
    f'Main Failure Factor: '
    f'{data["main_failure_factor"]}'
)


# ============================================================
# GRAPH 1
# ============================================================

st.subheader(
    "📈 Actual vs Predicted Timeline"
)

image1 = Image.open(
    "actual_vs_predicted.png"
)

st.image(
    image1,
    use_container_width=True
)


# ============================================================
# GRAPH 2
# ============================================================

st.subheader(
    "⚙️ Feature Importance"
)

image2 = Image.open(
    "feature_importance.png"
)

st.image(
    image2,
    use_container_width=True
)


# ============================================================
# GRAPH 3
# ============================================================

st.subheader(
    "🧠 SHAP Explainability"
)

image3 = Image.open(
    "shap_summary.png"
)

st.image(
    image3,
    use_container_width=True
)


# ============================================================
# AUTO REFRESH
# ============================================================

time.sleep(REFRESH_TIME)

st.rerun()

