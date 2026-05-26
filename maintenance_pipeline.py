import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from project_paths import (
    DEFAULT_PREDICT_DATA,
    DEFAULT_TRAIN_DATA,
    MODEL_PATH,
    PREDICTIONS_PATH,
    RESULTS_DIR,
    RESULTS_JSON,
    ensure_project_dirs,
    resolve_input_path,
)
from project_templates import DASHBOARD_CODE

TARGET_CANDIDATES = (
    "actual_remaining_life",
    "remaining_useful_life",
    "remaining_useful_life_hours",
    "remaining_life",
    "rul",
)

LEAK_OR_GENERATED_COLUMNS = {
    "actual_remaining_life",
    "remaining_useful_life",
    "remaining_useful_life_hours",
    "remaining_life",
    "rul",
    "predicted_rul",
    "predicted_remaining_life",
    "actual_rul",
    "prediction_error",
    "prediction",
    "failure_risk",
    "failure_risk_score",
    "alert_level",
    "breakdown_label",
    "fault_label",
    "anomaly_label",
    "predicted_breakdown_time",
    "predicted_breakdown_in_cycles",
    "predicted_breakdown_in_hours",
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
        path = resolve_input_path(Path(value)) if value else default
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
    args.train_data = resolve_input_path(args.train_data)
    args.predict_data = resolve_input_path(args.predict_data)
    
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


def infer_rul_unit(target_col: str | None) -> str:
    if target_col and "hour" in normalize_name(target_col):
        return "hours"
    return "cycles"


def unit_label(rul_unit: str) -> str:
    return "hours" if rul_unit == "hours" else "cycles"


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


def create_model() -> object:
    return make_pipeline(StandardScaler(), Ridge(alpha=1.0))


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

    validation_model = create_model()
    validation_model.fit(X_train, y_train)
    validation_predictions = np.clip(validation_model.predict(X_valid), 0.0, 1.0) * target_max

    final_model = create_model()
    final_model.fit(X, y_normalized)

    bundle = {
        "model": final_model,
        "model_name": "ridge_regression",
        "feature_columns": list(X.columns),
        "sensor_columns": sensor_cols,
        "target_column": target_col,
        "target_unit": infer_rul_unit(target_col),
        "training_csv": str(train_csv),
        "target_max": target_max,
        "validation_metrics": regression_metrics(y_valid_raw, validation_predictions),
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


def data_quality_warnings(
    predict_df: pd.DataFrame,
    target_col: str | None,
    bundle: Dict[str, object],
    prediction_scale: float,
) -> List[str]:
    warnings = []

    if target_col:
        actual = pd.to_numeric(predict_df[target_col], errors="coerce")
        actual_max = float(actual.max()) if actual.notna().any() else 0.0
        if actual_max > prediction_scale * 5.0:
            warnings.append(
                "Prediction data RUL scale is much larger than the model scale. "
                "Use a matching training dataset, or retrain the model for this dataset type."
            )

    prediction_sensors = {normalize_name(col) for col in detect_sensor_columns(predict_df, target_col)}
    training_sensors = {normalize_name(col) for col in bundle.get("sensor_columns", [])}
    if training_sensors:
        overlap = prediction_sensors & training_sensors
        overlap_ratio = len(overlap) / len(training_sensors)
        if overlap_ratio < 0.5:
            warnings.append(
                "Prediction sensor columns do not match the sensor columns used to train the model. "
                "The graph may look wrong because the model is being applied to a different dataset format."
            )

    return warnings


def format_duration(total_seconds: float) -> str:
    total_seconds = int(max(0.0, total_seconds))
    days, rem = divmod(total_seconds, 24 * 3600)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{days}d {hours}h {minutes}m {seconds}s"


def format_rul_duration(remaining_life: float, rul_unit: str, seconds_per_cycle: float) -> str:
    if rul_unit == "hours":
        return format_duration(remaining_life * 3600)
    return format_duration(remaining_life * seconds_per_cycle)


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
    rul_unit: str,
    seconds_per_cycle: float,
    early_warning_ratio: float,
) -> Dict[str, object]:
    warning_threshold = max(20.0, scale * early_warning_ratio)
    failure_threshold = max(5.0, scale * 0.02)

    output_df["Predicted_RUL"] = predictions.round(3)
    output_df["Predicted_RUL_Unit"] = unit_label(rul_unit)
    if rul_unit == "hours":
        output_df["Predicted_Breakdown_In_Hours"] = predictions.round(3)
    else:
        output_df["Predicted_Breakdown_In_Cycles"] = predictions.round(3)
    output_df["Predicted_Breakdown_Time"] = predictions.apply(
        lambda value: format_rul_duration(value, rul_unit, seconds_per_cycle)
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
            format_rul_duration(float(predictions.iloc[warning_index]), rul_unit, seconds_per_cycle)
            if warning_index is not None
            else None
        ),
        "predicted_failure_time_from_that_row": (
            format_rul_duration(float(predictions.iloc[failure_index]), rul_unit, seconds_per_cycle)
            if failure_index is not None
            else None
        ),
    }


def disallowed_model_features(bundle: Dict[str, object]) -> List[str]:
    disallowed = []
    for col in bundle.get("feature_columns", []):
        normalized = normalize_name(str(col))
        if any(
            normalized == blocked or normalized.startswith(f"{blocked}_")
            for blocked in LEAK_OR_GENERATED_COLUMNS
        ):
            disallowed.append(str(col))
    return disallowed


def training_path_from_bundle(bundle: Dict[str, object]) -> Path | None:
    value = bundle.get("training_csv")
    if not value:
        return None
    path = resolve_input_path(Path(str(value)))
    return path if path.exists() else None


def cycle_axis(df: pd.DataFrame, length: int) -> Tuple[np.ndarray, str]:
    cycle_candidates = (
        "cycle",
        "cycles",
        "machine_cycle",
        "cycle_number",
        "timestamp",
        "time",
    )
    normalized = {normalize_name(col): col for col in df.columns}

    for candidate in cycle_candidates:
        col = normalized.get(candidate)
        if col is None:
            continue
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().all() and values.nunique() > 1:
            return values.to_numpy(dtype=float), f"{col} (cycles)"

    return np.arange(length, dtype=float), "Machine cycle"


def smooth_line(
    x_values: np.ndarray,
    series: pd.Series,
    window: int = 5,
) -> Tuple[np.ndarray, np.ndarray]:
    smoothed = series.reset_index(drop=True).rolling(window, min_periods=1).mean()
    y = pd.to_numeric(smoothed, errors="coerce").ffill().bfill().fillna(0.0).to_numpy()
    x = np.asarray(x_values, dtype=float)

    if len(y) < 4 or np.unique(y).size < 2 or np.unique(x).size != len(x):
        return x, y

    dense_x = np.linspace(x.min(), x.max(), max(len(y) * 8, 120))
    try:
        dense_y = make_interp_spline(x, y, k=3)(dense_x)
        return dense_x, np.clip(dense_y, 0.0, None)
    except ValueError:
        return x, y


def save_actual_vs_predicted_plot(
    output_path: Path,
    source_df: pd.DataFrame,
    predictions: pd.Series,
    actual: pd.Series | None,
    timeline: Dict[str, object],
) -> None:
    plt.figure(figsize=(14, 6))
    x_values, x_label = cycle_axis(source_df, len(predictions))
    if actual is not None:
        actual_x, actual_y = smooth_line(x_values, actual)
        plt.plot(
            actual_x,
            actual_y,
            label="Actual RUL",
            linewidth=2.4,
        )
    predicted_x, predicted_y = smooth_line(x_values, predictions)
    plt.plot(
        predicted_x,
        predicted_y,
        label="Predicted RUL",
        linewidth=2.4,
        linestyle="--",
    )
    if timeline["early_warning_row"] is not None:
        warning_cycle = x_values[timeline["early_warning_row"]]
        plt.axvline(
            warning_cycle,
            color="#e0a100",
            linestyle=":",
            linewidth=2,
            label="Early warning",
        )
    if timeline["predicted_failure_row"] is not None:
        failure_cycle = x_values[timeline["predicted_failure_row"]]
        plt.axvline(
            failure_cycle,
            color="#c62828",
            linestyle=":",
            linewidth=2,
            label="Predicted breakdown",
        )
    plt.title("Actual vs Predicted Breakdown Timeline")
    plt.xlabel(x_label)
    plt.ylabel("Remaining Useful Life (RUL in cycles)")
    plt.grid(True, alpha=0.28)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def clean_generated_files() -> None:
    stale_paths = (
        Path("actual_vs_predicted.png"),
        Path("feature_importance.png"),
        RESULTS_DIR / "actual_vs_predicted.png",
        RESULTS_DIR / "feature_importance.png",
        RESULTS_DIR / "feature_importance.csv",
    )
    for path in stale_paths:
        if path.exists():
            path.unlink()
    if Path("__pycache__").exists():
        shutil.rmtree("__pycache__")
    ensure_project_dirs()


def write_dashboard() -> None:
    Path("dashboard.py").write_text(DASHBOARD_CODE, encoding="utf-8")


def run_prediction(args: argparse.Namespace) -> None:
    ensure_project_dirs()
    args = fill_interactive_args(args)
    if not args.no_clean:
        clean_generated_files()

    # Check if model exists and load it, or train if it doesn't exist
    if args.model_path.exists() and not args.force_retrain:
        print("\n[OK] Loading existing trained model...")
        bundle = joblib.load(args.model_path)
        resolved_training_path = training_path_from_bundle(bundle)
        if resolved_training_path is not None:
            bundle["training_csv"] = str(resolved_training_path)
        print(f"   Model trained on: {bundle['training_csv']}")
        blocked_features = disallowed_model_features(bundle)
        if blocked_features:
            print(
                f"   Found {len(blocked_features)} generated/label features in the saved model. "
                "Retraining from sensor columns only..."
            )
            retrain_path = args.train_data or training_path_from_bundle(bundle) or DEFAULT_TRAIN_DATA
            if not retrain_path.exists():
                retrain_path = prompt_path("Enter historical training CSV path", DEFAULT_TRAIN_DATA)
            args.train_data = retrain_path
            bundle = train_model(args.train_data, args.model_path)
    else:
        if args.force_retrain and args.model_path.exists():
            print("\n[INFO] Force retraining model (--force-retrain flag set)...")
        else:
            print("\n[INFO] Training new model from the given historical dataset...")
        
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
    warnings_list = data_quality_warnings(predict_df, target_col, bundle, prediction_scale)
    rul_unit = bundle.get("target_unit") or infer_rul_unit(bundle.get("target_column"))
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
        rul_unit=rul_unit,
        seconds_per_cycle=args.seconds_per_cycle,
        early_warning_ratio=args.early_warning_ratio,
    )
    if actual is not None:
        output_df["Prediction_Error"] = (output_df["Predicted_RUL"] - actual).round(3)
    output_df.to_csv(PREDICTIONS_PATH, index=False)

    save_actual_vs_predicted_plot(
        RESULTS_DIR / "actual_vs_predicted.png",
        predict_df,
        predictions,
        actual,
        timeline,
    )
    write_dashboard()

    latest_rul = timeline["latest_predicted_rul"]
    machine_status = status_from_rul(latest_rul, prediction_scale, args.early_warning_ratio)

    results = {
        "machine_status": machine_status,
        "remaining_life": int(round(latest_rul)),
        "remaining_life_unit": unit_label(rul_unit),
        "approx_failure_time": format_rul_duration(latest_rul, rul_unit, args.seconds_per_cycle),
        "early_warning_row": timeline["early_warning_row"],
        "predicted_failure_row": timeline["predicted_failure_row"],
        "early_warning_threshold": timeline["warning_threshold_cycles"],
        "failure_threshold": timeline["failure_threshold_cycles"],
        "model_name": bundle["model_name"],
        "training_csv": str(args.train_data or bundle.get("training_csv")),
        "prediction_csv": str(args.predict_data),
        "prediction_scale": prediction_scale,
        "prediction_unit": unit_label(rul_unit),
        "validation_metrics": bundle["validation_metrics"],
        "prediction_metrics": prediction_metrics,
        "warnings": warnings_list,
    }
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n========== MACHINE FAILURE REPORT ==========\n")
    print(f"Current machine status     : {machine_status}")
    print(f"Breakdown time from latest : {results['approx_failure_time']}")
    for warning in warnings_list:
        print(f"Warning                  : {warning}")
    print(f"Actual vs predicted graph  : {RESULTS_DIR / 'actual_vs_predicted.png'}")
    print("Dashboard command          : streamlit run dashboard.py")


if __name__ == "__main__":
    run_prediction(parse_args())
