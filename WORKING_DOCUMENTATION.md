# Working Documentation

## What The Project Does

The project predicts machine Remaining Useful Life (RUL) from sensor data. It uses one Ridge Regression model, predicts RUL for a machine dataset, assigns alert levels, saves results, and shows the summary in Streamlit.

## Simple Run

```powershell
python app.py --predict-data datasets\machine_vibration_testing_dataset.csv --train-data datasets\machine_vibration_training_dataset.csv --seconds-per-cycle 5
streamlit run dashboard.py
```

## Simple Data Simulation

All sample dataset generation now goes through one file:

```powershell
python data_simulator.py few-days
python data_simulator.py healthy
python data_simulator.py long-life
```

Every simulator run uses a fresh random seed by default, saves a numbered copy such as `few_days_machine_dataset_1.csv`, and also refreshes the latest file such as `few_days_machine_dataset.csv`. Use `--seed 42` when you want repeatable data.

For unique timelines, use `--profile healthy`, `--profile watch`, `--profile warning`, or `--profile critical` with `few-days`.

## Code Layout

- `app.py`: tiny entry point. This is the file to show first when explaining the project.
- `maintenance_pipeline.py`: Ridge model training, prediction, metrics, charting, and output generation.
- `data_simulator.py`: common simulator for sample datasets.
- `dashboard.py`: dashboard UI.
- `project_paths.py`: common path constants.
- `project_templates.py`: dashboard template used when the pipeline refreshes `dashboard.py`.

## Prediction Flow

1. Read training and prediction CSV paths.
2. Load the saved model, or train a new one when needed.
3. Build rolling sensor features.
4. Predict RUL.
5. Add breakdown timeline and alert level columns.
6. Save prediction CSV, graph, and `results.json`.
7. Open `dashboard.py` with Streamlit.

## Generated Outputs

- `results.json`: dashboard metrics.
- `results/machine_failure_predictions.csv`: row-by-row predictions.
- `results/actual_vs_predicted.png`: RUL comparison chart.

## Alert Meaning

- `RUNNING`: predicted RUL is comfortably above the warning threshold.
- `EARLY WARNING`: predicted RUL is low enough to plan maintenance.
- `BREAKDOWN IMMINENT`: predicted RUL is near the failure threshold.

## Removed Components

SHAP, model comparison, failure risk, main failure factor, and feature-importance outputs were removed to keep the project easier to explain. The project now focuses on one clear model and one clear prediction graph.
