# NALCO Predictive Maintenance & Fault Analysis System

Python and Streamlit project for predicting machine Remaining Useful Life (RUL), generating maintenance alerts, and showing results in a dashboard.

## Main Files

- `app.py`: small entry point for running the prediction pipeline.
- `maintenance_pipeline.py`: Ridge model training, prediction, feature creation, metrics, and chart generation.
- `dashboard.py`: Streamlit dashboard that reads `results.json` and generated charts.
- `data_simulator.py`: common dataset simulator for all sample data modes.
- `project_paths.py`: shared project paths.

## Install

```powershell
pip install -r requirements.txt
```

## Run Prediction

```powershell
python app.py --predict-data datasets\machine_vibration_testing_dataset.csv --train-data datasets\machine_vibration_training_dataset.csv --seconds-per-cycle 5
```

Use `--force-retrain` when you want to rebuild `machine_breakdown_model.pkl`.

## Run Dashboard

```powershell
streamlit run dashboard.py
```

## Generate Sample Data

Use the single simulator command:

```powershell
python data_simulator.py few-days
python data_simulator.py healthy
python data_simulator.py long-life
```

Each simulator run now saves two files inside `datasets/`:

- a numbered history file, such as `few_days_machine_dataset_1.csv`
- a latest file, such as `few_days_machine_dataset.csv`, which is the easy file to pass into `app.py`

By default, each run uses a new random seed. Use `--seed 42` when you want repeatable data.

For more different prediction timelines, use a few-days profile:

```powershell
python data_simulator.py few-days --profile healthy
python data_simulator.py few-days --profile watch
python data_simulator.py few-days --profile warning
python data_simulator.py few-days --profile critical
```

## Outputs

- `results.json`: dashboard summary values.
- `results/machine_failure_predictions.csv`: row-by-row predictions.
- `results/actual_vs_predicted.png`: actual vs predicted RUL graph.
