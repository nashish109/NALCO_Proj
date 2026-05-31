from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from ml.prediction_engine import PredictionEngine
from reports.maintenance_report import generate_maintenance_pdf
from simulation.future_degradation import simulate_future_degradation as simulate_future


def _num(data: dict[str, Any], *names: str, default: float = 0.0) -> float:
    for name in names:
        if name in data and data[name] not in ("", None):
            return float(data[name])
    return default


def predict_rul(sensor_window: pd.DataFrame, engine: PredictionEngine) -> dict[str, Any]:
    return {"tool": "predict_rul", "ok": True, "result": engine.predict_rul(sensor_window).to_json()}


def detect_anomaly(sensor_data: dict[str, Any]) -> dict[str, Any]:
    vibration = _num(sensor_data, "vibration_rms_mm_s", "vibration_sensor", default=2.0)
    temperature = _num(sensor_data, "temperature_c", "temperature_sensor", default=45.0)
    wear = _num(sensor_data, "bearing_wear_percent", "bearing_wear_sensor", default=5.0)
    lubricant = _num(sensor_data, "lubricant_quality_percent", default=95.0)

    score = (
        0.38 * np.clip(vibration / 8.8, 0.0, 1.0)
        + 0.24 * np.clip(temperature / 90.0, 0.0, 1.0)
        + 0.24 * np.clip(wear / 100.0, 0.0, 1.0)
        + 0.14 * np.clip(1.0 - lubricant / 100.0, 0.0, 1.0)
    )
    severity = "nominal"
    if score >= 0.78:
        severity = "critical"
    elif score >= 0.62:
        severity = "high"
    elif score >= 0.42:
        severity = "watch"

    return {
        "tool": "detect_anomaly",
        "ok": True,
        "result": {
            "anomaly_score": round(float(score), 3),
            "severity": severity,
            "is_anomaly": bool(score >= 0.42),
            "signals": {
                "vibration": vibration,
                "temperature": temperature,
                "bearing_wear": wear,
                "lubricant_quality": lubricant,
            },
        },
    }


def calculate_failure_probability(
    rul_hours: float,
    anomaly_score: float,
    sensor_data: dict[str, Any],
) -> dict[str, Any]:
    vibration = _num(sensor_data, "vibration_rms_mm_s", "vibration_sensor", default=2.0)
    temperature = _num(sensor_data, "temperature_c", "temperature_sensor", default=45.0)
    rul_pressure = 1.0 - np.clip(rul_hours / 195.0, 0.0, 1.0)
    thermal_pressure = np.clip((temperature - 55.0) / 35.0, 0.0, 1.0)
    vibration_pressure = np.clip((vibration - 3.0) / 5.8, 0.0, 1.0)
    probability = 0.46 * rul_pressure + 0.34 * anomaly_score + 0.12 * vibration_pressure + 0.08 * thermal_pressure
    return {
        "tool": "calculate_failure_probability",
        "ok": True,
        "result": {"failure_probability": round(float(np.clip(probability, 0.0, 1.0)), 3)},
    }


def generate_failure_analysis(
    sensor_data: dict[str, Any],
    rul_hours: float,
    anomaly: dict[str, Any],
    failure_probability: float,
) -> dict[str, Any]:
    signals = anomaly["result"]["signals"]
    reasons = []
    if signals["vibration"] >= 5.2:
        reasons.append("RMS vibration is crossing bearing instability bands.")
    if signals["temperature"] >= 68.0:
        reasons.append("Thermal load is rising with the degradation curve.")
    if signals["bearing_wear"] >= 55.0:
        reasons.append("Bearing wear has entered the accelerated fatigue region.")
    if signals["lubricant_quality"] <= 45.0:
        reasons.append("Lubricant quality is no longer damping mechanical friction.")
    if not reasons:
        reasons.append("Telemetry remains inside the stable operating envelope.")

    return {
        "tool": "generate_failure_analysis",
        "ok": True,
        "result": {
            "summary": " | ".join(reasons),
            "primary_failure_mode": "bearing wear and vibration harmonics",
            "rul_hours": round(rul_hours, 2),
            "failure_probability": round(failure_probability, 3),
            "evidence": reasons,
        },
    }


def recommend_maintenance(
    rul_hours: float,
    anomaly_score: float,
    failure_probability: float,
) -> dict[str, Any]:
    if rul_hours <= 8 or failure_probability >= 0.82:
        action, priority = "Emergency shutdown required", "RED"
    elif rul_hours <= 24 or failure_probability >= 0.66:
        action, priority = "Replace bearing and reduce operational load", "ORANGE"
    elif rul_hours <= 72 or anomaly_score >= 0.42:
        action, priority = "Schedule inspection within next shift", "YELLOW"
    else:
        action, priority = "Continue monitoring", "GREEN"

    return {
        "tool": "recommend_maintenance",
        "ok": True,
        "result": {
            "action": action,
            "priority": priority,
            "rationale": f"Decision based on RUL={rul_hours:.1f}h, anomaly={anomaly_score:.2f}, risk={failure_probability:.2f}.",
        },
    }


def send_alert(level: str, message: str) -> dict[str, Any]:
    return {
        "tool": "send_alert",
        "ok": True,
        "result": {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        },
    }


def simulate_future_degradation(
    sensor_data: dict[str, Any],
    rul_hours: float,
    failure_probability: float,
) -> dict[str, Any]:
    return {"tool": "simulate_future_degradation", "ok": True, "result": simulate_future(sensor_data, rul_hours, failure_probability)}


def generate_report(machine_state: dict[str, Any]) -> dict[str, Any]:
    path = generate_maintenance_pdf(machine_state)
    return {"tool": "generate_report", "ok": True, "result": {"pdf_path": str(path)}}
