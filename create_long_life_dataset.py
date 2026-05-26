import pandas as pd
import numpy as np
from pathlib import Path


DATASETS_DIR = Path("datasets")
DATASETS_DIR.mkdir(exist_ok=True)

# Set random seed for reproducibility
np.random.seed(42)

# Create training dataset with extended RUL range (0-150,000 cycles)
n_samples = 3000
timestamps = np.arange(n_samples)

# Vibration increases over time (degradation pattern)
vibration = 0.1 + (timestamps / n_samples) * 5.0 + np.random.normal(0, 0.2, n_samples)
vibration = np.clip(vibration, 0.05, 8.0)

temperature = 40 + np.random.normal(0, 2, n_samples)
pressure = 100 + np.random.normal(0, 3, n_samples)
rpm = 1500 + np.random.normal(0, 50, n_samples)
bearing_wear = 1 + (timestamps / n_samples) * 8.0 + np.random.normal(0, 0.3, n_samples)
bearing_wear = np.clip(bearing_wear, 0.5, 10.0)

# RUL decreases from 150,000 to 0 (linear degradation with some noise)
actual_rul = 150000 - (timestamps / n_samples) * 150000
actual_rul = actual_rul + np.random.normal(0, 500, n_samples)
actual_rul = np.clip(actual_rul, 0, 150000)

health_status = ['Healthy'] * n_samples

train_df = pd.DataFrame({
    'timestamp': timestamps,
    'vibration_sensor': vibration,
    'temperature_sensor': temperature,
    'pressure_sensor': pressure,
    'rpm_sensor': rpm,
    'bearing_wear_sensor': bearing_wear,
    'health_status': health_status,
    'actual_remaining_life': actual_rul
})

extended_training_path = DATASETS_DIR / 'extended_training_dataset.csv'
train_df.to_csv(extended_training_path, index=False)
print(f"[OK] Created {extended_training_path}")
print(f"   Samples: {len(train_df)}")
print(f"   RUL range: {train_df['actual_remaining_life'].min():.0f} - {train_df['actual_remaining_life'].max():.0f}")
print(f"   Vibration range: {train_df['vibration_sensor'].min():.3f} - {train_df['vibration_sensor'].max():.3f}")

# Create test dataset (moderately healthy machine = predict 180-200 hours)
n_test = 500

# Moderately low vibration (machine showing some age but still healthy)
vibration_test = 0.5 + np.random.normal(0, 0.1, n_test)
vibration_test = np.clip(vibration_test, 0.3, 1.0)

temperature_test = 45 + np.random.normal(0, 1.5, n_test)
pressure_test = 102 + np.random.normal(0, 2, n_test)
rpm_test = 1500 + np.random.normal(0, 40, n_test)
bearing_wear_test = 1.5 + np.random.normal(0, 0.15, n_test)
bearing_wear_test = np.clip(bearing_wear_test, 1.0, 2.5)

# Expected RUL in the 130,000-140,000 range (180-200 hours at 5 sec/cycle)
rul_test = 135000 + np.random.normal(0, 2000, n_test)
rul_test = np.clip(rul_test, 130000, 140000)

test_df = pd.DataFrame({
    'timestamp': np.arange(n_test),
    'vibration_sensor': vibration_test,
    'temperature_sensor': temperature_test,
    'pressure_sensor': pressure_test,
    'rpm_sensor': rpm_test,
    'bearing_wear_sensor': bearing_wear_test,
    'health_status': ['Healthy'] * n_test,
    'actual_remaining_life': rul_test
})

long_life_path = DATASETS_DIR / 'long_life_machine_dataset.csv'
test_df.to_csv(long_life_path, index=False)
print(f"\n[OK] Created {long_life_path}")
print(f"   Samples: {len(test_df)}")
print(f"   Expected RUL range: {test_df['actual_remaining_life'].min():.0f} - {test_df['actual_remaining_life'].max():.0f}")

breakdown_hours = test_df['actual_remaining_life'].iloc[0] * 5 / 3600
print(f"   Expected breakdown time (with 5 sec/cycle): {breakdown_hours:.1f} hours")
print(f"   Vibration range: {test_df['vibration_sensor'].min():.3f} - {test_df['vibration_sensor'].max():.3f}")
