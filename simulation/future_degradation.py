from __future__ import annotations

from typing import Any

import numpy as np


def simulate_future_degradation(
    sensor_data: dict[str, Any],
    rul_hours: float,
    failure_probability: float,
) -> dict[str, Any]:
    """Project sensor drift at 24h intervals until the likely failure horizon."""

    vibration = float(sensor_data.get("vibration_rms_mm_s", sensor_data.get("vibration_sensor", 2.0)))
    temperature = float(sensor_data.get("temperature_c", sensor_data.get("temperature_sensor", 45.0)))
    wear = float(sensor_data.get("bearing_wear_percent", sensor_data.get("bearing_wear_sensor", 8.0)))
    lubricant = float(sensor_data.get("lubricant_quality_percent", 85.0))

    points = []
    for hours in (0, 24, 48, 72):
        progress = min(1.0, hours / max(rul_hours, 1.0))
        acceleration = progress ** 1.45
        health = float(np.clip((rul_hours - hours) / max(rul_hours, 1.0) * 100.0, 0.0, 100.0))
        risk = float(np.clip(failure_probability + acceleration * 0.45, 0.0, 1.0))
        points.append(
            {
                "label": "Current" if hours == 0 else f"+{hours}h",
                "hours_from_now": hours,
                "health_percent": round(health, 2),
                "vibration_rms_mm_s": round(vibration * (1.0 + acceleration * 0.55), 3),
                "temperature_c": round(temperature + acceleration * 18.0, 2),
                "bearing_wear_percent": round(min(100.0, wear + acceleration * 38.0), 2),
                "lubricant_quality_percent": round(max(0.0, lubricant - acceleration * 32.0), 2),
                "failure_probability": round(risk, 3),
            }
        )

    failure_point = {
        "label": "Failure point",
        "hours_from_now": round(max(rul_hours, 0.0), 2),
        "health_percent": 0.0,
        "vibration_rms_mm_s": round(vibration * 1.72, 3),
        "temperature_c": round(temperature + 24.0, 2),
        "bearing_wear_percent": 100.0,
        "lubricant_quality_percent": max(0.0, round(lubricant - 45.0, 2)),
        "failure_probability": 1.0,
    }
    points.append(failure_point)
    return {"timeline": points, "estimated_death_timeline_hours": round(rul_hours, 2)}
