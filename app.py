import argparse
import json
import shutil
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


MODEL_PATH = Path("machine_breakdown_model.pkl")
RESULTS_DIR = Path("results")
PREDICTIONS_PATH = Path("machine_failure_predictions.csv")
RESULTS_JSON = Path("results.json")

DEFAULT_TRAIN_DATA = Path("simulated_training_machine_dataset.csv")
DEFAULT_PREDICT_DATA = Path("simulated_testing_machine_dataset.csv")

TARGET_CANDIDATES = (
    "actual_remaining_life",
    "remaining_useful_life",
    "remaining_life",
    "rul",
)

LEAK_OR_GENERATED_COLUMNS = {
    "actual_remaining_life",
    "remaining_useful_life",
    "remaining_life",
    "rul",
    "predicted_rul",
    "predicted_remaining_life",
    "actual_rul",
    "prediction_error",
    "prediction",
    "failure_risk",
    "alert_level",
    "predicted_breakdown_time",
    "predicted_breakdown_in_cycles",
    "timeline",
    "rms",
    "rollingmean",
    "rollingstd",
    "rms_diff",
    "ema",
}

TIME_COLUMNS = {"timestamp", "time", "date", "datetime"}
ROLLING_WINDOWS = (5, 15, 30)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive machine breakdown timeline predictor."
    )
    parser.add_argument("--train-data", type=Path, help="Training CSV path.")
    parser.add_argument("--predict-data", type=Path, help="New machine CSV path.")
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--seconds-per-cycle", type=float)
    parser.add_argument(
        "--rul-scale",
        type=float,
        help="Maximum possible RUL for an unlabeled prediction CSV.",
    )
    parser.add_argument(
        "--early-warning-ratio",
        type=float,
        default=0.12,
        help="Warning is raised when predicted RUL is below this share of the RUL scale.",
    )
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP plot generation for a faster run.",
    )
    parser.add_argument(
        "--no-clean",
        action="store_true",
        help="Do not remove stale generated files before running.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining of the model even if it already exists.",
    )
    return parser.parse_args()


def prompt_path(label: str, default: Path | None = None) -> Path:
    suffix = f" [{default}]" if default else ""
    while True:
        value = input(f"{label}{suffix}: ").strip().strip('"')
        path = Path(value) if value else default
        if path and path.exists():
            return path
        print("File not found. Please enter a valid CSV path.")


def prompt_float(label: str, default: float) -> float:
    while True:
        value = input(f"{label} [{default}]: ").strip()
        if not value:
            return default
        try:
            return float(value)
        except ValueError:
            print("Please enter a number.")


def fill_interactive_args(args: argparse.Namespace) -> argparse.Namespace:
    print("\n========== MACHINE BREAKDOWN PREDICTOR ==========\n")
    
    # Only ask for training data if model doesn't exist or force-retrain is set
    if args.train_data is None:
        if not args.model_path.exists() or args.force_retrain:
            args.train_data = prompt_path("Enter historical training CSV path", DEFAULT_TRAIN_DATA)
        # else: model exists and will be loaded, so we don't need training data
    
    # Always ask for prediction data
    if args.predict_data is None:
        args.predict_data = prompt_path("Enter new machine CSV path", DEFAULT_PREDICT_DATA)
    
    # Always ask for seconds per cycle
    if args.seconds_per_cycle is None:
        args.seconds_per_cycle = prompt_float("Enter seconds per machine cycle", 5.0)
    
    return args


def normalize_name(name: str) -> str:
    return name.strip().lower()


def find_target_column(df: pd.DataFrame) -> str:
    normalized = {normalize_name(col): col for col in df.columns}
    for candidate in TARGET_CANDIDATES:
        if candidate in normalized:
            return normalized[candidate]
    raise ValueError(
        "No RUL target column found. Add one of these columns to the training data: "
        + ", ".join(TARGET_CANDIDATES)
    )


def optional_target_column(df: pd.DataFrame) -> str | None:
    try:
        return find_target_column(df)
    except ValueError:
        return None


def detect_sensor_columns(df: pd.DataFrame, target_col: str | None) -> List[str]:
    excluded = set(LEAK_OR_GENERATED_COLUMNS)
    if target_col:
        excluded.add(normalize_name(target_col))

    numeric_columns = []
    for col in df.columns:
        name = normalize_name(col)
        if name in excluded or name in TIME_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)

    sensor_named = [col for col in numeric_columns if "sensor" in normalize_name(col)]
    return sensor_named or numeric_columns


def build_features(
    df: pd.DataFrame,
    *,
    target_col: str | None,
    training_columns: List[str] | None = None,
) -> Tuple[pd.DataFrame, List[str]]:
    parts = []
    sensor_cols = detect_sensor_columns(df, target_col)
    if not sensor_cols:
        raise ValueError("No numeric sensor columns found in the CSV.")

    for col in sensor_cols:
        series = pd.to_numeric(df[col], errors="coerce")
        feature_block = {
            col: series,
            f"{col}_diff": series.diff(),
            f"{col}_ema_10": series.ewm(span=10, adjust=False).mean(),
        }
        for window in ROLLING_WINDOWS:
            rolled = series.rolling(window=window, min_periods=1)
            feature_block[f"{col}_mean_{window}"] = rolled.mean()
            feature_block[f"{col}_std_{window}"] = rolled.std()
            feature_block[f"{col}_min_{window}"] = rolled.min()
            feature_block[f"{col}_max_{window}"] = rolled.max()
        parts.append(pd.DataFrame(feature_block, index=df.index))

    categorical_cols = [
        col
        for col in df.columns
        if col != target_col
        and normalize_name(col) not in LEAK_OR_GENERATED_COLUMNS
        and df[col].dtype == "object"
    ]
    if categorical_cols:
        parts.append(
            pd.get_dummies(
                df[categorical_cols].astype("string").fillna("missing"),
                prefix=categorical_cols,
                dummy_na=False,
            )
        )

    features = pd.concat(parts, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.ffill().bfill().fillna(0.0).astype(float)

    if training_columns is not None:
        features = features.reindex(columns=training_columns, fill_value=0.0)

    return features, sensor_cols


def time_split(
    X: pd.DataFrame,
    y: pd.Series,
    validation_fraction: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    split_at = int(len(X) * (1.0 - validation_fraction))
    split_at = max(1, min(split_at, len(X) - 1))
    return X.iloc[:split_at], X.iloc[split_at:], y.iloc[:split_at], y.iloc[split_at:]


def candidate_models() -> Dict[str, object]:
    return {
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=450,
            learning_rate=0.04,
            max_leaf_nodes=31,
            l2_regularization=0.05,
            random_state=42,
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=450,
            max_features=0.85,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1,
        ),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
    }


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_model(train_csv: Path, model_path: Path) -> Dict[str, object]:
    train_df = pd.read_csv(train_csv)
    target_col = find_target_column(train_df)
    X, sensor_cols = build_features(train_df, target_col=target_col)

    y_raw = pd.to_numeric(train_df[target_col], errors="coerce")
    valid_mask = y_raw.notna()
    X = X.loc[valid_mask].reset_index(drop=True)
    y_raw = y_raw.loc[valid_mask].reset_index(drop=True)
    target_max = float(max(y_raw.max(), 1.0))
    y_normalized = y_raw / target_max

    X_train, X_valid, y_train, _ = time_split(X, y_normalized)
    _, _, _, y_valid_raw = time_split(X, y_raw)

    scores = {}
    fitted = {}
    for name, model in candidate_models().items():
        model.fit(X_train, y_train)
        predicted_rul = np.clip(model.predict(X_valid), 0.0, 1.0) * target_max
        scores[name] = regression_metrics(y_valid_raw, predicted_rul)
        fitted[name] = model

    best_name = min(scores, key=lambda key: scores[key]["mae"])
    best_model = fitted[best_name]
    validation_predictions = np.clip(best_model.predict(X_valid), 0.0, 1.0) * target_max

    final_model = candidate_models()[best_name]
    final_model.fit(X, y_normalized)

    bundle = {
        "model": final_model,
        "model_name": best_name,
        "feature_columns": list(X.columns),
        "sensor_columns": sensor_cols,
        "target_column": target_col,
        "training_csv": str(train_csv),
        "target_max": target_max,
        "validation_metrics": regression_metrics(y_valid_raw, validation_predictions),
        "candidate_metrics": scores,
    }
    joblib.dump(bundle, model_path)
    return bundle


def infer_prediction_scale(
    args: argparse.Namespace,
    predict_df: pd.DataFrame,
    target_col: str | None,
    bundle: Dict[str, object],
) -> float:
    if args.rul_scale is not None:
        return float(max(args.rul_scale, 1.0))
    # Always use the training target_max as the scale for consistent unnormalization
    training_scale = bundle.get("target_max", 1.0)
    if training_scale > 1.0:
        return float(training_scale)
    if target_col:
        target = pd.to_numeric(predict_df[target_col], errors="coerce")
        if target.notna().any():
            return float(max(target.max(), 1.0))
    return float(max(len(predict_df), 1.0))


def format_duration(cycles: float, seconds_per_cycle: float) -> str:
    total_seconds = int(max(0.0, cycles) * seconds_per_cycle)
    days, rem = divmod(total_seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"


def status_from_rul(remaining_life: float, scale: float, early_warning_ratio: float) -> str:
    failure_at = max(5.0, scale * 0.02)
    warning_at = max(20.0, scale * early_warning_ratio)
    if remaining_life <= failure_at:
        return "BREAKDOWN IMMINENT"
    if remaining_life <= warning_at:
        return "EARLY WARNING"
    return "RUNNING"


def add_breakdown_timeline(
    output_df: pd.DataFrame,
    predictions: pd.Series,
    *,
    scale: float,
    seconds_per_cycle: float,
    early_warning_ratio: float,
) -> Dict[str, object]:
    warning_threshold = max(20.0, scale * early_warning_ratio)
    failure_threshold = max(5.0, scale * 0.02)

    output_df["Predicted_RUL"] = predictions.round(3)
    output_df["Predicted_Breakdown_In_Cycles"] = predictions.round(3)
    output_df["Predicted_Breakdown_Time"] = predictions.apply(
        lambda value: format_duration(value, seconds_per_cycle)
    )
    output_df["Alert_Level"] = predictions.apply(
        lambda value: status_from_rul(value, scale, early_warning_ratio)
    )

    warning_rows = output_df.index[predictions <= warning_threshold].tolist()
    failure_rows = output_df.index[predictions <= failure_threshold].tolist()

    warning_index = int(warning_rows[0]) if warning_rows else None
    failure_index = int(failure_rows[0]) if failure_rows else None
    
    # Use RUL at failure point if available, otherwise use last non-zero, otherwise last value
    if failure_index is not None:
        latest_prediction = float(predictions.iloc[failure_index])
    else:
        non_zero = predictions[predictions > 0.1]
        latest_prediction = float(non_zero.iloc[-1]) if len(non_zero) > 0 else float(predictions.iloc[-1])

    return {
        "latest_predicted_rul": latest_prediction,
        "warning_threshold_cycles": round(warning_threshold, 3),
        "failure_threshold_cycles": round(failure_threshold, 3),
        "early_warning_row": warning_index,
        "predicted_failure_row": failure_index,
        "early_warning_time_from_that_row": (
            format_duration(float(predictions.iloc[warning_index]), seconds_per_cycle)
            if warning_index is not None
            else None
        ),
        "predicted_failure_time_from_that_row": (
            format_duration(float(predictions.iloc[failure_index]), seconds_per_cycle)
            if failure_index is not None
            else None
        ),
    }


def feature_importance(bundle: Dict[str, object], X: pd.DataFrame, y: pd.Series | None) -> pd.DataFrame:
    model = bundle["model"]
    columns = bundle["feature_columns"]
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "named_steps") and "ridge" in model.named_steps:
        values = np.abs(np.asarray(model.named_steps["ridge"].coef_, dtype=float))
    elif y is not None and y.notna().sum() >= 10:
        sample_size = min(250, len(X))
        result = permutation_importance(
            model,
            X.tail(sample_size),
            y.tail(sample_size),
            n_repeats=6,
            random_state=42,
            n_jobs=-1,
        )
        values = result.importances_mean
    else:
        values = np.zeros(len(columns), dtype=float)
    return pd.DataFrame({"feature": columns, "importance": values}).sort_values(
        "importance", ascending=False
    )


def save_actual_vs_predicted_plot(
    output_path: Path,
    predictions: pd.Series,
    actual: pd.Series | None,
    timeline: Dict[str, object],
) -> None:
    plt.figure(figsize=(14, 6))
    if actual is not None:
        plt.plot(
            actual.reset_index(drop=True).rolling(5, min_periods=1).mean(),
            label="Actual RUL",
            linewidth=2.4,
        )
    plt.plot(
        predictions.reset_index(drop=True).rolling(5, min_periods=1).mean(),
        label="Predicted RUL",
        linewidth=2.4,
        linestyle="--",
    )
    if timeline["early_warning_row"] is not None:
        plt.axvline(
            timeline["early_warning_row"],
            color="#e0a100",
            linestyle=":",
            linewidth=2,
            label="Early warning",
        )
    if timeline["predicted_failure_row"] is not None:
        plt.axvline(
            timeline["predicted_failure_row"],
            color="#c62828",
            linestyle=":",
            linewidth=2,
            label="Predicted breakdown",
        )
    plt.title("Actual vs Predicted Breakdown Timeline")
    plt.xlabel("Sample row")
    plt.ylabel("Remaining Useful Life")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_feature_importance_plot(output_path: Path, importance_df: pd.DataFrame) -> None:
    top = importance_df.head(15).sort_values("importance", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature"], top["importance"], color="#2457a6")
    plt.title("Top Sensor Factors Used by the Model")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_shap_plot(output_path: Path, bundle: Dict[str, object], X: pd.DataFrame) -> bool:
    try:
        import shap

        sample = X.tail(min(60, len(X)))
        model = bundle["model"]

        def predict_fn(values):
            frame = pd.DataFrame(values, columns=sample.columns)
            return model.predict(frame)

        explainer = shap.Explainer(predict_fn, sample)
        shap_values = explainer(sample, silent=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.summary_plot(shap_values, sample, show=False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return True
    except Exception as exc:
        print(f"SHAP plot skipped: {exc}")
        return False


def clean_generated_files() -> None:
    for path in (Path("actual_vs_predicted.png"), Path("feature_importance.png"), Path("shap_summary.png")):
        if path.exists():
            path.unlink()
    if Path("__pycache__").exists():
        shutil.rmtree("__pycache__")
    RESULTS_DIR.mkdir(exist_ok=True)


def write_dashboard() -> None:
    dashboard_code = '''import json
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
'''
    Path("dashboard.py").write_text(dashboard_code, encoding="utf-8")


def write_documentation() -> None:
    doc = """# Machine Breakdown Prediction - Working Documentation

## Purpose

This project trains a Remaining Useful Life (RUL) model from historical machine sensor data and then predicts the breakdown timeline for a new machine dataset.

The new workflow is intentionally interactive:

1. Run the program.
2. Enter the training CSV path.
3. The model trains from that CSV.
4. Enter the new machine CSV path.
5. The program predicts RUL, early-warning point, predicted breakdown point, plots, metrics, and dashboard files.

## Expected CSV Format

The training CSV must contain:

- Numeric sensor columns, preferably named like `vibration_sensor`, `temperature_sensor`, `pressure_sensor`, `rpm_sensor`, `bearing_wear_sensor`.
- One target column for actual remaining life. Supported names are:
  - `actual_remaining_life`
  - `remaining_useful_life`
  - `remaining_life`
  - `rul`

The prediction CSV should contain the same sensor columns. If it also contains actual RUL, the program calculates accuracy and draws actual vs predicted. If it does not contain actual RUL, the program still predicts the breakdown timeline and draws the predicted curve.

## How To Run

Install requirements:

```powershell
pip install -r requirements.txt
```

Interactive run:

```powershell
python app.py
```

Example non-interactive run:

```powershell
python app.py --train-data simulated_training_machine_dataset.csv --predict-data simulated_testing_machine_dataset.csv --seconds-per-cycle 5
```

For a prediction file that has no actual RUL column, provide the expected maximum RUL/cycle horizon:

```powershell
python app.py --train-data simulated_training_machine_dataset.csv --predict-data new_machine_data.csv --rul-scale 900
```

## Generated Files

- `machine_breakdown_model.pkl`: saved trained model bundle.
- `machine_failure_predictions.csv`: row-by-row predictions with alert levels.
- `results.json`: final machine status and metrics.
- `results/actual_vs_predicted.png`: graph comparing actual and predicted RUL when actual RUL exists.
- `results/feature_importance.png`: main sensor factors used by the model.
- `results/feature_importance.csv`: feature importance values.
- `results/shap_summary.png`: explainability plot when SHAP succeeds.
- `dashboard.py`: Streamlit dashboard.

## Dashboard

Run:

```powershell
streamlit run dashboard.py
```

Then open the local URL shown by Streamlit.

## Model Design

The model does not use timestamp or row number as a shortcut. It uses sensor values and rolling behavior:

- current sensor value
- sensor difference
- exponential moving average
- rolling mean
- rolling standard deviation
- rolling minimum
- rolling maximum

The target is normalized RUL, which lets the same trained pattern work on datasets with different maximum cycle ranges.

The program tests multiple models and selects the one with the lowest validation MAE:

- HistGradientBoostingRegressor
- ExtraTreesRegressor
- Ridge regression

## Early Prediction Logic

The program reports two timeline points:

- Early warning row: first row where predicted RUL falls below the warning threshold.
- Predicted breakdown row: first row where predicted RUL falls below the breakdown threshold.

By default, the early warning threshold is 12 percent of the prediction scale, with a minimum of 20 cycles. This is intentionally earlier than the final breakdown point so maintenance can be planned before failure.

## Notes

Accuracy should be close, not artificially perfect. The model avoids leaked output columns such as `Predicted_RUL`, `Actual_RUL`, `RMS`, `EMA`, and old generated columns from earlier runs.
"""
    Path("WORKING_DOCUMENTATION.md").write_text(doc, encoding="utf-8")


def run_prediction(args: argparse.Namespace) -> None:
    args = fill_interactive_args(args)
    if not args.no_clean:
        clean_generated_files()

    # Check if model exists and load it, or train if it doesn't exist
    if args.model_path.exists() and not args.force_retrain:
        print("\n✅ Loading existing trained model...")
        bundle = joblib.load(args.model_path)
        print(f"   Model trained on: {bundle['training_csv']}")
    else:
        if args.force_retrain and args.model_path.exists():
            print("\n🔄 Force retraining model (--force-retrain flag set)...")
        else:
            print("\n📚 Training new model from the given historical dataset...")
        
        # Ensure train_data is provided for training
        if args.train_data is None:
            args.train_data = prompt_path("Enter historical training CSV path", DEFAULT_TRAIN_DATA)
        
        bundle = train_model(args.train_data, args.model_path)

    predict_df = pd.read_csv(args.predict_data)
    target_col = optional_target_column(predict_df)
    X, _ = build_features(
        predict_df,
        target_col=target_col,
        training_columns=bundle["feature_columns"],
    )

    prediction_scale = infer_prediction_scale(args, predict_df, target_col, bundle)
    predictions = pd.Series(
        np.clip(bundle["model"].predict(X), 0.0, 1.0) * prediction_scale,
        name="Predicted_RUL",
    )

    actual = None
    prediction_metrics = None
    if target_col:
        actual = pd.to_numeric(predict_df[target_col], errors="coerce")
        valid = actual.notna()
        if valid.any():
            prediction_metrics = regression_metrics(actual[valid], predictions[valid])

    output_df = predict_df.copy()
    timeline = add_breakdown_timeline(
        output_df,
        predictions,
        scale=prediction_scale,
        seconds_per_cycle=args.seconds_per_cycle,
        early_warning_ratio=args.early_warning_ratio,
    )
    if actual is not None:
        output_df["Prediction_Error"] = (output_df["Predicted_RUL"] - actual).round(3)
    output_df.to_csv(PREDICTIONS_PATH, index=False)

    importance_df = feature_importance(bundle, X, actual)
    importance_df.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)
    main_factor = str(importance_df.iloc[0]["feature"]) if not importance_df.empty else "unknown"

    save_actual_vs_predicted_plot(
        RESULTS_DIR / "actual_vs_predicted.png",
        predictions,
        actual,
        timeline,
    )
    save_feature_importance_plot(RESULTS_DIR / "feature_importance.png", importance_df)
    shap_created = False if args.skip_shap else save_shap_plot(RESULTS_DIR / "shap_summary.png", bundle, X)
    write_dashboard()
    write_documentation()

    latest_rul = timeline["latest_predicted_rul"]
    machine_status = status_from_rul(latest_rul, prediction_scale, args.early_warning_ratio)
    failure_risk = max(0.0, min(100.0, 100.0 * (1.0 - latest_rul / max(prediction_scale, 1.0))))

    results = {
        "machine_status": machine_status,
        "remaining_life": int(round(latest_rul)),
        "failure_risk": round(failure_risk, 2),
        "main_failure_factor": main_factor,
        "approx_failure_time": format_duration(latest_rul, args.seconds_per_cycle),
        "early_warning_row": timeline["early_warning_row"],
        "predicted_failure_row": timeline["predicted_failure_row"],
        "early_warning_threshold_cycles": timeline["warning_threshold_cycles"],
        "failure_threshold_cycles": timeline["failure_threshold_cycles"],
        "model_name": bundle["model_name"],
        "training_csv": str(args.train_data),
        "prediction_csv": str(args.predict_data),
        "prediction_scale": prediction_scale,
        "validation_metrics": bundle["validation_metrics"],
        "prediction_metrics": prediction_metrics,
        "shap_created": shap_created,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n========== MACHINE FAILURE REPORT ==========\n")
    print(f"Training data (model)       : {bundle['training_csv']}")
    print(f"Prediction data             : {args.predict_data}")
    print(f"Selected model              : {bundle['model_name']}")
    print(f"Current machine status     : {machine_status}")
    print(f"Latest predicted RUL       : {int(round(latest_rul))} cycles")
    print(f"Breakdown time from latest : {results['approx_failure_time']}")
    print(f"Early warning row          : {timeline['early_warning_row']}")
    print(f"Predicted breakdown row    : {timeline['predicted_failure_row']}")
    print(f"Main sensor factor         : {main_factor}")
    print(f"Failure risk               : {failure_risk:.2f}%")
    print("\nValidation metrics from training split:")
    for key, value in bundle["validation_metrics"].items():
        print(f"  {key.upper():4}: {value:.3f}")
    if prediction_metrics:
        print("\nActual vs predicted metrics on new data:")
        for key, value in prediction_metrics.items():
            print(f"  {key.upper():4}: {value:.3f}")
    print(f"\nSaved predictions          : {PREDICTIONS_PATH}")
    print(f"Saved graph                : {RESULTS_DIR / 'actual_vs_predicted.png'}")
    print(f"Saved documentation        : WORKING_DOCUMENTATION.md")
    print("Dashboard command          : streamlit run dashboard.py")


if __name__ == "__main__":
    run_prediction(parse_args())
