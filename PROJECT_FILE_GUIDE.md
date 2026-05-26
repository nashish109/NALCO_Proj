# Machine Fault Detection Project Guide

This guide explains the current simplified project from start to end in plain language.

## 1. What This Project Does

The project predicts **Remaining Useful Life**, also called **RUL**, for a machine.

RUL means:

```text
How much useful working time is left before the machine reaches failure condition?
```

The project reads machine sensor data such as vibration, temperature, RPM, acoustic noise, bearing wear, and lubricant quality. It then predicts the remaining life and shows the result in a Streamlit dashboard.

The dashboard now focuses only on the most important values:

- Machine status
- Approximate breakdown time
- Remaining life
- Model name
- Actual vs predicted graph

Failure risk and main failure factor were removed to keep the project simple and easier to explain.

## 2. Complete Working Flow

```text
Generate or use machine sensor data
        |
        v
Run app.py
        |
        v
Load or train one Ridge Regression model
        |
        v
Predict Remaining Useful Life
        |
        v
Create results.json, prediction CSV, and graph
        |
        v
Open dashboard.py with Streamlit
```

## 3. How To Run From Start

Open PowerShell:

```powershell
cd "E:\NALCO Proj\Machine_Fault_detection_by_SensorData"
```

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Install packages if needed:

```powershell
pip install -r requirements.txt
```

Generate sample data:

```powershell
python data_simulator.py few-days
```

Train once and predict:

```powershell
python app.py --train-data datasets\machine_vibration_training_dataset.csv --predict-data datasets\few_days_machine_dataset.csv --seconds-per-cycle 5 --force-retrain
```

Open the dashboard:

```powershell
streamlit run dashboard.py
```

For normal testing after the model is already trained, do not use `--force-retrain`:

```powershell
python app.py --predict-data datasets\few_days_machine_dataset.csv --seconds-per-cycle 5
```

## 4. Important Rule About Training

You do not need to train every time.

Use this only when you want a fresh model:

```powershell
--force-retrain
```

Use this for direct testing with the saved model:

```powershell
python app.py --predict-data datasets\few_days_machine_dataset.csv --seconds-per-cycle 5
```

The saved model file is:

```text
machine_breakdown_model.pkl
```

## 5. Model Used

The project now uses only one model:

```text
Ridge Regression
```

In code, it is created like this:

```python
make_pipeline(StandardScaler(), Ridge(alpha=1.0))
```

This means two things happen:

1. `StandardScaler()` makes sensor values easier for the model to compare.
2. `Ridge()` learns the relationship between sensor features and remaining useful life.

Simple explanation:

```text
Sensor readings go in.
Remaining useful life comes out.
```

Why Ridge Regression is used:

- It was performing well in this project.
- It is fast.
- It is stable.
- It is easier to explain than multiple competing models.
- It keeps the project simple without reducing practical result quality for this dataset.

## 6. Project Files

```text
app.py
maintenance_pipeline.py
data_simulator.py
dashboard.py
project_paths.py
project_templates.py
debug_predictions.py
requirements.txt
machine_breakdown_model.pkl
results.json
datasets/
results/
```

## 7. `app.py`

This is the entry point.

It is intentionally very small:

```python
from maintenance_pipeline import parse_args, run_prediction

if __name__ == "__main__":
    run_prediction(parse_args())
```

Meaning:

- Read command-line inputs.
- Start the prediction workflow.
- Keep heavy logic outside `app.py`.

This makes the project easier to explain.

## 8. `maintenance_pipeline.py`

This is the main working file.

It handles:

- input arguments
- model training
- saved model loading
- feature creation
- prediction
- alert status
- graph creation
- result file creation

### Important Constants

`TARGET_CANDIDATES`

Possible names for the actual RUL column:

```text
actual_remaining_life
remaining_useful_life
remaining_useful_life_hours
remaining_life
rul
```

The code checks these names to find the correct target column.

`LEAK_OR_GENERATED_COLUMNS`

Columns that should not be used as sensor inputs.

Examples:

```text
actual_remaining_life
predicted_rul
prediction_error
alert_level
breakdown_label
```

This prevents the model from cheating by learning from answer columns.

`ROLLING_WINDOWS`

```text
5, 15, 30
```

These are used to create rolling sensor features.

### Important Functions

`parse_args()`

Reads command-line options such as:

- `--train-data`
- `--predict-data`
- `--seconds-per-cycle`
- `--force-retrain`

`fill_interactive_args()`

Fills missing values using defaults or user input.

`find_target_column()`

Finds the actual RUL column in the training dataset.

`detect_sensor_columns()`

Chooses real numeric sensor columns and ignores answer/generated columns.

`build_features()`

Creates model input features from raw sensor readings.

For each sensor, it creates:

- current value
- rolling mean
- rolling standard deviation
- difference from previous row
- exponential moving average

This helps the model understand not only the current sensor value, but also recent behavior.

`create_model()`

Creates the single Ridge Regression model:

```python
make_pipeline(StandardScaler(), Ridge(alpha=1.0))
```

`train_model()`

Trains the Ridge model.

Steps:

1. Read training CSV.
2. Find actual RUL column.
3. Build sensor features.
4. Split data into training and validation parts.
5. Train Ridge Regression.
6. Check validation accuracy.
7. Train final model on full training data.
8. Save it to `machine_breakdown_model.pkl`.

`infer_prediction_scale()`

Finds the RUL scale used to convert model output back into real hours or cycles.

`data_quality_warnings()`

Checks whether the prediction dataset matches the training dataset format.

This helps catch mistakes such as testing long-life data with the normal hours-based model.

`status_from_rul()`

Converts predicted RUL into status:

```text
RUNNING
EARLY WARNING
BREAKDOWN IMMINENT
```

`add_breakdown_timeline()`

Adds prediction columns to the output CSV:

- `Predicted_RUL`
- `Predicted_RUL_Unit`
- `Predicted_Breakdown_Time`
- `Alert_Level`

`save_actual_vs_predicted_plot()`

Creates:

```text
results/actual_vs_predicted.png
```

`clean_generated_files()`

Deletes old generated files before creating fresh output.

`run_prediction()`

Runs the full pipeline:

1. Prepare folders.
2. Load or train model.
3. Read prediction data.
4. Build features.
5. Predict RUL.
6. Save prediction CSV.
7. Save graph.
8. Write `results.json`.
9. Print terminal report.

## 9. `data_simulator.py`

This file creates sample machine datasets.

Commands:

```powershell
python data_simulator.py few-days
python data_simulator.py healthy
python data_simulator.py long-life
```

Each run uses a fresh random seed by default, then saves a numbered file and a latest file.

Use a fixed seed only when you want repeatable demo data:

```powershell
python data_simulator.py few-days --seed 42
```

Use profiles when you want visibly different prediction timelines:

```powershell
python data_simulator.py few-days --profile healthy
python data_simulator.py few-days --profile watch
python data_simulator.py few-days --profile warning
python data_simulator.py few-days --profile critical
```

Profile meaning:

- `healthy`: high remaining life, mostly running.
- `watch`: medium remaining life, useful for normal decline demos.
- `warning`: lower remaining life, should move into early warning.
- `critical`: very low remaining life, should move near breakdown imminent.

Example for `few-days`:

```text
datasets/few_days_machine_dataset_1.csv
datasets/few_days_machine_dataset.csv
```

Use the numbered file when you want to compare different generated datasets. Use the latest file when you want the simplest command for prediction.

`few-days`

Creates a dataset where the machine has a few days of remaining life.

Output examples:

```text
datasets/few_days_machine_dataset_1.csv
datasets/few_days_machine_dataset.csv
```

`healthy`

Creates a healthier machine sample.

Output examples:

```text
datasets/healthy_machine_dataset_1.csv
datasets/healthy_machine_dataset.csv
```

`long-life`

Creates a separate long-life training and testing dataset.

Output examples:

```text
datasets/extended_training_dataset_1.csv
datasets/extended_training_dataset.csv
datasets/long_life_machine_dataset_1.csv
datasets/long_life_machine_dataset.csv
```

Important:

Do not test `long_life_machine_dataset.csv` with the normal `machine_vibration_training_dataset.csv` model. The scale and sensor columns are different.

Correct long-life command:

```powershell
python data_simulator.py long-life
python app.py --train-data datasets\extended_training_dataset.csv --predict-data datasets\long_life_machine_dataset.csv --seconds-per-cycle 5 --force-retrain
```

## 10. `dashboard.py`

This is the Streamlit dashboard.

It reads:

```text
results.json
```

and displays:

- machine status
- breakdown time
- remaining life
- model name
- actual vs predicted graph

If `results.json` does not exist, it asks the user to run `python app.py` first.

## 11. `project_paths.py`

This stores common paths:

- project root
- `datasets/`
- `results/`
- `machine_breakdown_model.pkl`
- `results.json`
- prediction CSV path
- default training data
- default prediction data

This avoids repeating paths across the project.

## 12. `project_templates.py`

This stores dashboard code as a template.

The pipeline can refresh `dashboard.py` from this template.

## 13. `debug_predictions.py`

This is a helper script for checking predictions.

Run it after `app.py`:

```powershell
python debug_predictions.py
```

It prints:

- prediction file path
- number of rows
- minimum predicted RUL
- maximum predicted RUL
- mean predicted RUL
- first 20 prediction rows

It is useful for debugging, but it is not required for normal dashboard use.

## 14. Input Files

Main training dataset:

```text
datasets/machine_vibration_training_dataset.csv
```

Main prediction dataset:

```text
datasets/machine_vibration_testing_dataset.csv
```

Generated few-days dataset:

```text
datasets/few_days_machine_dataset_1.csv
datasets/few_days_machine_dataset.csv
```

## 15. Output Files

`results.json`

Summary values for dashboard.

`results/machine_failure_predictions.csv`

Full row-by-row prediction table.

`results/actual_vs_predicted.png`

Graph comparing actual RUL and predicted RUL.

## 16. Presentation Explanation

Use this simple explanation:

```text
This project predicts machine remaining useful life from sensor readings.
We use one Ridge Regression model to keep the system simple and explainable.
The model learns from historical sensor data where actual remaining life is known.
After training, it predicts remaining life for new machine readings.
The dashboard shows status, breakdown time, remaining life, model name, and the prediction graph.
```

## 17. Final Short Summary

```text
data_simulator.py       creates sample data
app.py                  starts prediction
maintenance_pipeline.py contains the main Ridge model logic
dashboard.py            shows results
```

One-line summary:

```text
The project uses machine sensor data and a Ridge Regression model to predict remaining useful life and show the result in a dashboard.
```
