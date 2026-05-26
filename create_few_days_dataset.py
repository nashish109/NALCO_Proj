import numpy as np
import pandas as pd
from pathlib import Path


np.random.seed(42)

DATASETS_DIR = Path("datasets")
OUTPUT_PATH = DATASETS_DIR / "few_days_machine_dataset.csv"
N_SAMPLES = 240
MAX_TRAINING_RUL_HOURS = 195.0

timestamps = pd.date_range("2026-01-02 00:00:00", periods=N_SAMPLES, freq="30min")

# This dataset represents a moderately worn machine with 3-4 days of useful life.
remaining_useful_life_hours = np.linspace(96.0, 72.0, N_SAMPLES)
remaining_useful_life_hours += np.random.normal(0, 1.2, N_SAMPLES)
remaining_useful_life_hours = np.clip(remaining_useful_life_hours, 72.0, 96.0)

degradation = 1.0 - (remaining_useful_life_hours / MAX_TRAINING_RUL_HOURS)

vibration_rms_mm_s = 1.8 + degradation * 6.5 + np.random.normal(0, 0.18, N_SAMPLES)
temperature_c = 42.0 + degradation * 45.0 + np.random.normal(0, 1.4, N_SAMPLES)
rpm = 1500.0 - degradation * 165.0 + np.random.normal(0, 8.0, N_SAMPLES)
acoustic_noise_db = 55.0 + degradation * 30.0 + np.random.normal(0, 0.9, N_SAMPLES)
bearing_wear_percent = degradation * 100.0 + np.random.normal(0, 1.8, N_SAMPLES)
lubricant_quality_percent = 100.0 - degradation * 75.0 + np.random.normal(
    0, 1.6, N_SAMPLES
)

vibration_rms_mm_s = np.clip(vibration_rms_mm_s, 1.5, 8.8)
temperature_c = np.clip(temperature_c, 40.0, 90.0)
rpm = np.clip(rpm, 1330.0, 1510.0)
acoustic_noise_db = np.clip(acoustic_noise_db, 55.0, 85.0)
bearing_wear_percent = np.clip(bearing_wear_percent, 0.0, 100.0)
lubricant_quality_percent = np.clip(lubricant_quality_percent, 24.0, 100.0)

failure_risk_score = (
    0.35 * (vibration_rms_mm_s / 8.8)
    + 0.25 * (bearing_wear_percent / 100.0)
    + 0.20 * (1.0 - lubricant_quality_percent / 100.0)
    + 0.20 * (temperature_c / 90.0)
)
failure_risk_score = np.clip(failure_risk_score, 0.0, 1.0)

df = pd.DataFrame(
    {
        "timestamp": timestamps,
        "vibration_rms_mm_s": vibration_rms_mm_s.round(3),
        "temperature_c": temperature_c.round(2),
        "rpm": rpm.round(1),
        "acoustic_noise_db": acoustic_noise_db.round(2),
        "bearing_wear_percent": bearing_wear_percent.round(2),
        "lubricant_quality_percent": lubricant_quality_percent.round(2),
        "failure_risk_score": failure_risk_score.round(4),
        "remaining_useful_life_hours": remaining_useful_life_hours.round(2),
        "breakdown_label": 0,
    }
)

DATASETS_DIR.mkdir(exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"[OK] Created {OUTPUT_PATH}")
print(f"Samples: {len(df)}")
print(
    "Actual RUL range: "
    f"{df['remaining_useful_life_hours'].min():.1f} - "
    f"{df['remaining_useful_life_hours'].max():.1f} hours"
)
print(
    "Actual RUL range: "
    f"{df['remaining_useful_life_hours'].min() / 24:.1f} - "
    f"{df['remaining_useful_life_hours'].max() / 24:.1f} days"
)
print(
    "Vibration range: "
    f"{df['vibration_rms_mm_s'].min():.2f} - "
    f"{df['vibration_rms_mm_s'].max():.2f} mm/s"
)
