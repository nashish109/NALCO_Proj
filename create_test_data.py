import pandas as pd
import numpy as np

# Create a "healthier" test dataset with lower vibration values
df = pd.read_csv('simulated_testing_machine_dataset.csv').copy()

# Reduce vibration to simulate a healthier machine
df['vibration_sensor'] = df['vibration_sensor'] * 0.3

# Adjust the actual RUL accordingly (machine has longer life)
df['actual_remaining_life'] = 1400

df.to_csv('healthy_machine_dataset.csv', index=False)
print("✅ Created healthy_machine_dataset.csv")
print(f"Vibration range: {df['vibration_sensor'].min():.3f} - {df['vibration_sensor'].max():.3f}")
print(f"Expected RUL: {df['actual_remaining_life'].iloc[0]}")
