# Machine Breakdown Prediction - Working Documentation

## Purpose

This project trains a Remaining Useful Life (RUL) model from historical machine sensor data and then predicts the breakdown timeline for a new machine dataset.

The new workflow is intentionally interactive:

1. Run the program.
2. Enter the training CSV path.
3. The model trains from that CSV.
4. Enter the new machine CSV path.
5. The program predicts RUL, early-warning point, predicted breakdown point, plots, metrics, and dashboard files.

## Expected CSV Format

The training CSV must contain:

- Numeric sensor columns, preferably named like `vibration_sensor`, `temperature_sensor`, `pressure_sensor`, `rpm_sensor`, `bearing_wear_sensor`.
- One target column for actual remaining life. Supported names are:
  - `actual_remaining_life`
  - `remaining_useful_life`
  - `remaining_life`
  - `rul`

The prediction CSV should contain the same sensor columns. If it also contains actual RUL, the program calculates accuracy and draws actual vs predicted. If it does not contain actual RUL, the program still predicts the breakdown timeline and draws the predicted curve.

## How To Run

Install requirements:

```powershell
pip install -r requirements.txt
```

Interactive run:

```powershell
python app.py
```

Example non-interactive run:

```powershell
python app.py --train-data simulated_training_machine_dataset.csv --predict-data simulated_testing_machine_dataset.csv --seconds-per-cycle 5
```

For a prediction file that has no actual RUL column, provide the expected maximum RUL/cycle horizon:

```powershell
python app.py --train-data simulated_training_machine_dataset.csv --predict-data new_machine_data.csv --rul-scale 900
```

## Generated Files

- `machine_breakdown_model.pkl`: saved trained model bundle.
- `machine_failure_predictions.csv`: row-by-row predictions with alert levels.
- `results.json`: final machine status and metrics.
- `results/actual_vs_predicted.png`: graph comparing actual and predicted RUL when actual RUL exists.
- `results/feature_importance.png`: main sensor factors used by the model.
- `results/feature_importance.csv`: feature importance values.
- `results/shap_summary.png`: explainability plot when SHAP succeeds.
- `dashboard.py`: Streamlit dashboard.

## Dashboard

Run:

```powershell
streamlit run dashboard.py
```

Then open the local URL shown by Streamlit.

## Model Design

The model does not use timestamp or row number as a shortcut. It uses sensor values and rolling behavior:

- current sensor value
- sensor difference
- exponential moving average
- rolling mean
- rolling standard deviation
- rolling minimum
- rolling maximum

The target is normalized RUL, which lets the same trained pattern work on datasets with different maximum cycle ranges.

The program tests multiple models and selects the one with the lowest validation MAE:

- HistGradientBoostingRegressor
- ExtraTreesRegressor
- Ridge regression

## Early Prediction Logic

The program reports two timeline points:

- Early warning row: first row where predicted RUL falls below the warning threshold.
- Predicted breakdown row: first row where predicted RUL falls below the breakdown threshold.

By default, the early warning threshold is 12 percent of the prediction scale, with a minimum of 20 cycles. This is intentionally earlier than the final breakdown point so maintenance can be planned before failure.

## Notes

Accuracy should be close, not artificially perfect. The model avoids leaked output columns such as `Predicted_RUL`, `Actual_RUL`, `RMS`, `EMA`, and old generated columns from earlier runs.
