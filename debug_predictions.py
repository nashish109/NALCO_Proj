import pandas as pd
import numpy as np

df = pd.read_csv('machine_failure_predictions.csv')
print(f'Min: {df["Predicted_RUL"].min()}')
print(f'Max: {df["Predicted_RUL"].max()}')
print(f'Mean: {df["Predicted_RUL"].mean()}')
print(f'Non-zero count: {(df["Predicted_RUL"] > 0).sum()} / {len(df)}')
print(f'\nFirst 20 rows:')
print(df[['Predicted_RUL', 'actual_remaining_life']].head(20))
