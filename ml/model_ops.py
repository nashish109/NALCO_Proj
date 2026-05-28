from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import joblib
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from maintenance_pipeline import (
    build_features,
    find_target_column,
    optional_target_column,
    regression_metrics,
    train_model,
)
from project_paths import DEFAULT_TRAIN_DATA, MODEL_PATH


def retrain_rul_model(train_csv: Path = DEFAULT_TRAIN_DATA, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    bundle = train_model(train_csv, model_path)
    return {
        "model_path": str(model_path),
        "training_csv": str(train_csv),
        "model_name": bundle.get("model_name", "ridge_regression"),
        "target_column": bundle.get("target_column"),
        "target_unit": bundle.get("target_unit"),
        "target_max": bundle.get("target_max"),
        "validation_metrics": bundle.get("validation_metrics", {}),
    }


def evaluate_model(test_csv: Path, model_path: Path = MODEL_PATH) -> dict[str, Any]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        bundle = joblib.load(model_path)
    df = pd.read_csv(test_csv)
    target_col = optional_target_column(df) or find_target_column(df)
    features, _ = build_features(
        df,
        target_col=target_col,
        training_columns=bundle["feature_columns"],
    )
    actual = pd.to_numeric(df[target_col], errors="coerce")
    predictions = pd.Series(bundle["model"].predict(features), name="prediction")
    predictions = predictions.clip(0.0, 1.0) * float(bundle.get("target_max", 1.0))
    valid = actual.notna()
    metrics = regression_metrics(actual[valid], predictions[valid])
    return {
        "test_csv": str(test_csv),
        "target_column": target_col,
        "samples": int(valid.sum()),
        "metrics": metrics,
        "credibility_notes": [
            "Evaluation is only credible when the test file represents real unseen machine history.",
            "Synthetic simulator data is useful for demos and stress tests, but should not be treated as production evidence.",
        ],
    }
