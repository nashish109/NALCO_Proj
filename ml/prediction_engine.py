from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import InconsistentVersionWarning

from maintenance_pipeline import (
    build_features,
    optional_target_column,
    train_model,
)
from project_paths import DEFAULT_TRAIN_DATA, MODEL_PATH


@dataclass
class PredictionResult:
    rul_hours: float
    health_percent: float
    model_name: str
    confidence: float
    sensor_columns: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "rul_hours": round(self.rul_hours, 2),
            "health_percent": round(self.health_percent, 2),
            "model_name": self.model_name,
            "confidence": round(self.confidence, 3),
            "sensor_columns": self.sensor_columns,
        }


class PredictionEngine:
    """Loads the Ridge RUL model and exposes JSON-friendly predictions."""

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        train_data: Path = DEFAULT_TRAIN_DATA,
    ) -> None:
        self.model_path = model_path
        self.train_data = train_data
        self.bundle = self._load_or_train()

    def _load_or_train(self) -> dict[str, Any]:
        if self.model_path.exists():
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", InconsistentVersionWarning)
                    return joblib.load(self.model_path)
            except Exception:
                return train_model(self.train_data, self.model_path)
        return train_model(self.train_data, self.model_path)

    @property
    def target_max(self) -> float:
        return float(max(self.bundle.get("target_max", 195.0), 1.0))

    def predict_rul(self, frame: pd.DataFrame) -> PredictionResult:
        target_col = optional_target_column(frame)
        features, sensor_columns = build_features(
            frame,
            target_col=target_col,
            training_columns=self.bundle["feature_columns"],
        )
        raw_prediction = self.bundle["model"].predict(features.tail(1))[0]
        rul_hours = float(np.clip(raw_prediction, 0.0, 1.0) * self.target_max)
        health_percent = float(np.clip((rul_hours / self.target_max) * 100.0, 0.0, 100.0))

        validation = self.bundle.get("validation_metrics", {}) or {}
        r2 = float(validation.get("r2", 0.86))
        confidence = float(np.clip(0.72 + max(r2, 0.0) * 0.24, 0.55, 0.98))

        return PredictionResult(
            rul_hours=rul_hours,
            health_percent=health_percent,
            model_name=str(self.bundle.get("model_name", "ridge_regression")),
            confidence=confidence,
            sensor_columns=sensor_columns,
        )
