# FINAL INDUSTRIAL THRESHOLD-BASED VERSION
# Google Colab Compatible
# ============================================================


# ============================================================
# STEP 1 — INSTALL LIBRARIES
# ============================================================

#!pip install xgboost shap joblib -q


# ============================================================
# STEP 2 — IMPORT LIBRARIES
# ============================================================

import os
import joblib
import shap

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

#from google.colab import files

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# STEP 3 — CONFIGURATION
# ============================================================

MODEL_PATH = "machine_breakdown_model.pkl"

WINDOW_SIZE = 15

CRITICAL_VIBRATION = 8


# ============================================================
# STEP 4 — FEATURE ENGINEERING
# ============================================================

def create_features(df, vibration_col):

    df = df.copy()

    # Timeline
    df["timeline"] = np.arange(len(df))

    # RMS
    df["RMS"] = (
        df[vibration_col]
        .rolling(WINDOW_SIZE)
        .apply(lambda x: np.sqrt(np.mean(x**2)))
    )

    # Rolling Mean
    df["RollingMean"] = (
        df["RMS"]
        .rolling(WINDOW_SIZE)
        .mean()
    )

    # Rolling STD
    df["RollingSTD"] = (
        df["RMS"]
        .rolling(WINDOW_SIZE)
        .std()
    )

    # RMS Difference
    df["RMS_Diff"] = df["RMS"].diff()

    # EMA
    df["EMA"] = (
        df["RMS"]
        .ewm(span=WINDOW_SIZE)
        .mean()
    )

    # Remove invalid rows
    df.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    df.dropna(inplace=True)

    return df


# ============================================================
# STEP 5 — TRAIN MODEL ONLY ONCE
# ============================================================

if not os.path.exists(MODEL_PATH):

    print("\n📤 Upload HISTORICAL TRAINING DATASET")

    train_file = input("Enter training CSV path: ")

    train_df = pd.read_csv(train_file)

    print("\n✅ Historical Dataset Loaded")


    # ========================================================
    # DISPLAY COLUMNS
    # ========================================================

    print("\nColumns Available:\n")

    for i, col in enumerate(train_df.columns):

        print(f"{i} : {col}")


    # ========================================================
    # SELECT VIBRATION COLUMN
    # ========================================================

    col_index = int(
        input(
            "\nEnter vibration column index: "
        )
    )

    vibration_col = train_df.columns[col_index]

    print(
        f"\nUsing vibration column: "
        f"{vibration_col}"
    )


    # ========================================================
    # FEATURE ENGINEERING
    # ========================================================

    train_df = create_features(
        train_df,
        vibration_col
    )


    # ========================================================
    # THRESHOLD-BASED RUL
    # ========================================================

    train_df["RUL"] = (
        (
            CRITICAL_VIBRATION -
            train_df[vibration_col]
        ) * 40
    )

    # Prevent negative RUL
    train_df["RUL"] = (
        train_df["RUL"]
        .clip(lower=0)
    )


    # ========================================================
    # FEATURES
    # ========================================================

    FEATURES = [
        "RMS",
        "RollingMean",
        "RollingSTD",
        "RMS_Diff",
        "EMA"
    ]


    X = train_df[FEATURES]

    y = train_df["RUL"]


    # ========================================================
    # TRAIN TEST SPLIT
    # ========================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )


    # ========================================================
    # TRAIN MODEL
    # ========================================================

    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        random_state=42
    )

    model.fit(X_train, y_train)

    print("\n✅ Model Training Completed")


    # ========================================================
    # SAVE MODEL
    # ========================================================

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
            "vibration_col": vibration_col
        },
        MODEL_PATH
    )

    print("\n✅ Trained Model Saved Successfully")


    # ========================================================
    # EVALUATION
    # ========================================================

    predictions = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    print("\n========== TRAINING PERFORMANCE ==========\n")

    print(f"MAE  : {mae:.2f}")

    print(f"RMSE : {rmse:.2f}")

    print(f"R²   : {r2:.4f}")


    # ========================================================
    # TRAINING GRAPH
    # ========================================================

    train_graph = pd.DataFrame({

        "Actual_RUL":
            y_test.values,

        "Predicted_RUL":
            predictions

    })

    train_graph["Actual_RUL"] = (
        train_graph["Actual_RUL"]
        .rolling(5, min_periods=1)
        .mean()
    )

    train_graph["Predicted_RUL"] = (
        train_graph["Predicted_RUL"]
        .rolling(5, min_periods=1)
        .mean()
    )

    plt.figure(figsize=(16,6))

    plt.plot(
        train_graph["Actual_RUL"],
        label="Actual Remaining Life",
        linewidth=3
    )

    plt.plot(
        train_graph["Predicted_RUL"],
        label="Predicted Remaining Life",
        linewidth=3,
        linestyle="dashed"
    )

    plt.title(
        "Training Dataset: Actual vs Predicted RUL"
    )

    plt.xlabel("Timeline")

    plt.ylabel("Remaining Useful Life")

    plt.legend()

    plt.grid(True)

    plt.show()


else:

    print("\n✅ Existing Trained Model Found")

    saved_objects = joblib.load(
        MODEL_PATH
    )

    model = saved_objects["model"]

    FEATURES = saved_objects["features"]

    vibration_col = saved_objects["vibration_col"]


# ============================================================
# STEP 6 — UPLOAD NEW MACHINE DATASET
# ============================================================

print("\n📤 Upload NEW MACHINE DATASET")

new_file = input("Enter new machine CSV path: ")

new_df = pd.read_csv(new_file)

print("\n✅ New Machine Dataset Loaded")


# ============================================================
# STEP 7 — FEATURE ENGINEERING FOR NEW DATA
# ============================================================

new_df = create_features(
    new_df,
    vibration_col
)


# ============================================================
# STEP 8 — PREDICT REMAINING LIFE
# ============================================================

X_new = new_df[FEATURES]

X_new = X_new.dropna()

predictions = model.predict(X_new)

predictions = pd.Series(predictions)

predictions.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

predictions.dropna(inplace=True)

if len(predictions) == 0:

    raise ValueError(
        "Prediction failed. "
        "Dataset too small or invalid."
    )

# Smooth predictions
predictions = (
    predictions
    .rolling(5, min_periods=1)
    .mean()
)

new_df = new_df.iloc[-len(predictions):].copy()

new_df["Predicted_RUL"] = predictions.values


# ============================================================
# STEP 9 — CREATE ACTUAL RUL
# ============================================================

new_df["Actual_RUL"] = (
    (
        CRITICAL_VIBRATION -
        new_df[vibration_col]
    ) * 40
)

new_df["Actual_RUL"] = (
    new_df["Actual_RUL"]
    .clip(lower=0)
)


# ============================================================
# STEP 10 — FAILURE TIME PREDICTION
# ============================================================

latest_prediction = int(
    round(
        float(
            new_df["Predicted_RUL"].iloc[-1]
        )
    )
)

latest_prediction = max(
    0,
    latest_prediction
)

print("\n========== MACHINE FAILURE REPORT ==========\n")


# ============================================================
# MACHINE CYCLE TIME
# ============================================================

SECONDS_PER_CYCLE = float(
    input(
        "\nEnter seconds per machine cycle: "
    )
)


# ============================================================
# TOTAL REMAINING TIME
# ============================================================

total_seconds = int(
    latest_prediction *
    SECONDS_PER_CYCLE
)


# ============================================================
# TIME CONVERSION
# ============================================================

days = total_seconds // (24 * 3600)

remaining = total_seconds % (24 * 3600)

hours = remaining // 3600

remaining %= 3600

minutes = remaining // 60

seconds = remaining % 60


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    f"Machine Status           : ",
    end=""
)

if latest_prediction < 20:

    machine_status = "CRITICAL"

elif latest_prediction < 80:

    machine_status = "WARNING"

else:

    machine_status = "HEALTHY"

print(machine_status)

print(
    f"Estimated Remaining Life : "
    f"{latest_prediction} cycles"
)

failure_risk = (
    100 -
    (
        latest_prediction /
        (CRITICAL_VIBRATION * 40)
    ) * 100
)

failure_risk = max(
    0,
    min(100, failure_risk)
)

print(
    f"Failure Risk             : "
    f"{failure_risk:.2f}%"
)

print(
    f"\nApprox Failure Time      : "
    f"{days}d {hours}h {minutes}m {seconds}s"
)


# ============================================================
# STEP 11 — ACTUAL VS PREDICTED GRAPH
# ============================================================

comparison = pd.DataFrame({

    "Actual_RUL":
        new_df["Actual_RUL"],

    "Predicted_RUL":
        new_df["Predicted_RUL"]

})

comparison["Actual_RUL"] = (
    comparison["Actual_RUL"]
    .rolling(5, min_periods=1)
    .mean()
)

comparison["Predicted_RUL"] = (
    comparison["Predicted_RUL"]
    .rolling(5, min_periods=1)
    .mean()
)

plt.figure(figsize=(16,6))

plt.plot(
    comparison["Actual_RUL"],
    label="Actual Remaining Life",
    linewidth=3
)

plt.plot(
    comparison["Predicted_RUL"],
    label="Predicted Remaining Life",
    linewidth=3,
    linestyle="dashed"
)

plt.title(
    "Actual vs Predicted Failure Timeline"
)

plt.xlabel("Timeline")

plt.ylabel("Remaining Useful Life")

plt.legend()

plt.grid(True)

plt.show()


# ============================================================
# STEP 12 — FEATURE IMPORTANCE
# ============================================================

importance = model.feature_importances_

feature_df = pd.DataFrame({

    "Feature":
        FEATURES,

    "Importance":
        importance

})

feature_df = feature_df.sort_values(
    by="Importance",
    ascending=True
)

plt.figure(figsize=(8,4))

plt.barh(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.title(
    "Factors Affecting Machine Breakdown"
)

plt.xlabel("Importance")

plt.show()


# ============================================================
# STEP 13 — SHAP EXPLAINABILITY
# ============================================================

print("\nGenerating Explainable AI Analysis...")

sample_data = X_new.sample(
    min(200, len(X_new)),
    random_state=42
)

explainer = shap.Explainer(
    model,
    sample_data
)

shap_values = explainer(sample_data)

shap.summary_plot(
    shap_values,
    sample_data
)


# ============================================================
# STEP 14 — FAILURE REASON ANALYSIS
# ============================================================

top_feature = feature_df.iloc[-1]["Feature"]

print("\n========== POSSIBLE FAILURE REASONS ==========\n")

if top_feature == "RMS":

    print(
        "Primary reason: Increasing vibration RMS detected."
    )

elif top_feature == "RollingSTD":

    print(
        "Primary reason: High vibration instability detected."
    )

elif top_feature == "RollingMean":

    print(
        "Primary reason: Continuous degradation trend detected."
    )

elif top_feature == "RMS_Diff":

    print(
        "Primary reason: Sudden vibration spikes detected."
    )

elif top_feature == "EMA":

    print(
        "Primary reason: Long-term vibration degradation trend detected."
    )


# ============================================================
# STEP 15 — SAVE REPORT
# ============================================================

new_df.to_csv(
    "machine_failure_predictions.csv",
    index=False
)

print(
    "\n✅ Prediction report saved as:"
)

print(
    "machine_failure_predictions.csv"
)


# ============================================================
# STEP 16 — FINAL SUMMARY
# ============================================================

print("\n========== FINAL SUMMARY ==========\n")

print(f"Machine Status           : {machine_status}")

print(
    f"Estimated Remaining Life : "
    f"{latest_prediction} cycles"
)

print(
    f"Failure Risk             : "
    f"{failure_risk:.2f}%"
)

print(
    f"Main Failure Factor      : "
    f"{top_feature}"
)

print(
    f"\nApprox Failure Time      : "
    f"{days}d {hours}h {minutes}m {seconds}s"
)

# ============================================================
# STEP 17 — LIVE AI DASHBOARD WEBSITE
# ADD THIS ENTIRE BLOCK AT THE END OF YOUR ORIGINAL CODE
# ============================================================


# ============================================================
# INSTALL STREAMLIT
# ============================================================

#!pip install streamlit pillow -q


# ============================================================
# SAVE RESULTS JSON
# ============================================================

import json

results = {

    "machine_status":
        machine_status,

    "remaining_life":
        latest_prediction,

    "failure_risk":
        round(failure_risk, 2),

    "main_failure_factor":
        top_feature,

    "approx_failure_time":
        f"{days}d {hours}h "
        f"{minutes}m {seconds}s"
}

with open(
    "results.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f
    )


# ============================================================
# SAVE ACTUAL VS PREDICTED GRAPH
# ============================================================

plt.figure(figsize=(16,6))

plt.plot(
    comparison["Actual_RUL"],
    label="Actual Remaining Life",
    linewidth=3
)

plt.plot(
    comparison["Predicted_RUL"],
    label="Predicted Remaining Life",
    linewidth=3,
    linestyle="dashed"
)

plt.title(
    "Actual vs Predicted Failure Timeline"
)

plt.xlabel("Timeline")

plt.ylabel("Remaining Useful Life")

plt.legend()

plt.grid(True)

plt.savefig(
    "actual_vs_predicted.png"
)

plt.close()


# ============================================================
# SAVE FEATURE IMPORTANCE GRAPH
# ============================================================

plt.figure(figsize=(8,4))

plt.barh(
    feature_df["Feature"],
    feature_df["Importance"]
)

plt.title(
    "Factors Affecting Machine Breakdown"
)

plt.xlabel("Importance")

plt.savefig(
    "feature_importance.png"
)

plt.close()


# ============================================================
# SAVE SHAP GRAPH
# ============================================================

shap.summary_plot(
    shap_values,
    sample_data,
    show=False
)

plt.savefig(
    "shap_summary.png"
)

plt.close()


print(
    "\n✅ Dashboard files generated successfully"
)


# ============================================================
# CREATE STREAMLIT DASHBOARD
# ============================================================

dashboard_code = '''

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

'''


with open(
    "dashboard.py",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        dashboard_code
    )


print(
    "\n Streamlit dashboard created"
)


# ============================================================
# START STREAMLIT SERVER
# ============================================================

import subprocess
import threading
import time


def run_streamlit():

    subprocess.Popen(
        [
            "streamlit",
            "run",
            "dashboard.py",
            "--server.port",
            "8507"
        ]
    )


threading.Thread(
    target=run_streamlit
).start()


time.sleep(15)

print(
    "\n Streamlit server started"
)

# ============================================================
# CREATE PUBLIC WEBSITE URL
# ============================================================

#!wget -q -O cloudflared \
#https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64

#!chmod +x cloudflared


#print(
#    "\n Creating public dashboard URL..."
#)


#!./cloudflared tunnel --url http://localhost:8507

