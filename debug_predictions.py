import pandas as pd

from project_paths import PREDICTIONS_PATH
TARGET_CANDIDATES = (
    "actual_remaining_life",
    "remaining_useful_life",
    "remaining_useful_life_hours",
    "remaining_life",
    "rul",
)


if not PREDICTIONS_PATH.exists():
    raise FileNotFoundError("Run python app.py first to create results/machine_failure_predictions.csv")

df = pd.read_csv(PREDICTIONS_PATH)
actual_col = next((col for col in TARGET_CANDIDATES if col in df.columns), None)

print(f"Prediction file: {PREDICTIONS_PATH}")
print(f"Rows: {len(df)}")
print(f"Min RUL: {df['Predicted_RUL'].min():.3f}")
print(f"Max RUL: {df['Predicted_RUL'].max():.3f}")
print(f"Mean RUL: {df['Predicted_RUL'].mean():.3f}")
print(f"Non-zero count: {(df['Predicted_RUL'] > 0).sum()} / {len(df)}")

preview_cols = ["Predicted_RUL", "Predicted_RUL_Unit", "Predicted_Breakdown_Time", "Alert_Level"]
if actual_col:
    preview_cols.insert(0, actual_col)
    preview_cols.append("Prediction_Error")

print("\nFirst 20 rows:")
print(df[[col for col in preview_cols if col in df.columns]].head(20))
