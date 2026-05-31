from __future__ import annotations

from datetime import datetime
from typing import Any


CANONICAL_ALIASES = {
    "timestamp": ("timestamp", "time", "datetime"),
    "vibration_rms_mm_s": ("vibration_rms_mm_s", "vibration", "vibration_sensor", "vibration_mm_s"),
    "temperature_c": ("temperature_c", "temperature", "temperature_sensor", "bearing_temperature_c"),
    "rpm": ("rpm", "rpm_sensor", "motor_rpm"),
    "acoustic_noise_db": ("acoustic_noise_db", "noise_db", "sound_db"),
    "bearing_wear_percent": ("bearing_wear_percent", "bearing_wear", "bearing_wear_sensor"),
    "lubricant_quality_percent": ("lubricant_quality_percent", "lubricant_quality", "oil_quality_percent"),
    "remaining_useful_life_hours": ("remaining_useful_life_hours", "rul_hours", "actual_remaining_life"),
}


def canonicalize_telemetry(payload: dict[str, Any], tag_map: dict[str, str] | None = None) -> dict[str, Any]:
    """Convert real plant tags or simulator fields into the model's canonical schema."""

    tag_map = tag_map or {}
    canonical: dict[str, Any] = {}

    for canonical_name, source_name in tag_map.items():
        if source_name in payload:
            canonical[canonical_name] = payload[source_name]

    for canonical_name, aliases in CANONICAL_ALIASES.items():
        if canonical_name in canonical:
            continue
        for alias in aliases:
            if alias in payload:
                canonical[canonical_name] = payload[alias]
                break

    canonical.setdefault("timestamp", datetime.utcnow().isoformat(timespec="seconds") + "Z")
    return canonical


def telemetry_quality(payload: dict[str, Any]) -> dict[str, Any]:
    required = ("vibration_rms_mm_s", "temperature_c", "rpm")
    optional = ("acoustic_noise_db", "bearing_wear_percent", "lubricant_quality_percent")
    missing_required = [field for field in required if field not in payload]
    missing_optional = [field for field in optional if field not in payload]
    score = max(0.0, 1.0 - len(missing_required) * 0.25 - len(missing_optional) * 0.06)
    return {
        "score": round(score, 3),
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "ready_for_prediction": not missing_required,
    }
