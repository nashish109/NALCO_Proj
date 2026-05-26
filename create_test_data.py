import pandas as pd
import numpy as np
from pathlib import Path


DATASETS_DIR = Path("datasets")

# Create a "healthier" test dataset with lower vibration values
df = pd.read_csv(DATASETS_DIR / 'simulated_testing_machine_dataset.csv').copy()

# Reduce vibration to simulate a healthier machine
df['vibration_sensor'] = df['vibration_sensor'] * 0.3

# Adjust the actual RUL accordingly (machine has longer life)
df['actual_remaining_life'] = 1400

DATASETS_DIR.mkdir(exist_ok=True)
output_path = DATASETS_DIR / 'healthy_machine_dataset.csv'
df.to_csv(output_path, index=False)
print(f"[OK] Created {output_path}")
print(f"Vibration range: {df['vibration_sensor'].min():.3f} - {df['vibration_sensor'].max():.3f}")
print(f"Expected RUL: {df['actual_remaining_life'].iloc[0]}")
