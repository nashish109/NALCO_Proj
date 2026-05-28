# LAST 200 HOURS - Complete Project File Guide

Project tagline:

```text
Every machine whispers before it dies. Hermes Agent learns to listen.
```

This guide explains the complete current project after the Hermes Agent transformation. It replaces the older Streamlit/RUL-only guide.

The project is now an industrial AI monitoring platform with:

- live telemetry replay from CSV datasets
- external real-machine telemetry ingestion through API
- RUL prediction
- anomaly detection
- failure probability calculation
- Hermes Agent reasoning and maintenance decisions
- future degradation simulation
- alerting
- PDF maintenance reports
- enterprise-grade browser dashboard
- simulator-driven test data generation
- model evaluation and retraining endpoints

## 1. Current Project Purpose

The system predicts machine health and remaining useful life, then lets Hermes Agent act as the reasoning layer.

The old pipeline was:

```text
Sensor Data -> ML Prediction -> Dashboard
```

The current pipeline is:

```text
CSV Replay or Real Telemetry
        |
        v
Canonical Telemetry Schema
        |
        v
Prediction Engine + Anomaly Tools
        |
        v
Hermes Agent
        |
        +--> Analyze degradation
        +--> Decide maintenance action
        +--> Explain failure evidence
        +--> Simulate future machine state
        +--> Generate alerts/reports
        |
        v
Flask API + Enterprise Dashboard
```

## 2. Important Truth About The Data

By default, the dashboard is not connected to a real factory machine. It replays rows from:

```text
datasets/few_days_machine_dataset.csv
```

This replay is handled by:

```text
simulation/sensor_stream.py
```

Every frontend refresh calls:

```http
GET /api/state
```

That advances the telemetry stream by one row and returns a full Hermes Agent state.

The project also supports real telemetry through:

```http
POST /api/telemetry
```

So the system is now structured like a deployable industrial prototype: CSV replay for demo/testing, API ingestion for real plant data.

## 3. Quick Start

Open PowerShell in the project root:

```powershell
cd "E:\PERSONAL PROJECTS\INDUSTRIAL_AI_AGENT_SYSTEM"
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the application:

```powershell
python app.py
```

Open the dashboard:

```text
http://127.0.0.1:5000
```

Stop the server with `Ctrl+C` in the terminal if it is running in the foreground.

## 4. Main Runtime Flow

When the browser opens:

1. `frontend/index.html` loads the enterprise dashboard shell.
2. `frontend/app.js` starts polling `/api/state`.
3. `backend/hermes_api.py` receives the request.
4. `HermesAgent.tick()` advances one monitoring cycle.
5. `SensorStream.next_sample()` returns a telemetry row and a rolling window.
6. `PredictionEngine.predict_rul()` predicts remaining useful life.
7. Tool functions calculate anomaly score, risk, failure analysis, maintenance recommendation, alert, and future simulation.
8. Flask returns JSON to the browser.
9. The frontend updates KPI cards, charts, tables, alerts, logs, and settings status.

## 5. Current Folder Structure

```text
agent/
backend/
config/
datasets/
frontend/
ingestion/
ml/
reports/
simulation/
tools/
app.py
data_simulator.py
maintenance_pipeline.py
project_paths.py
machine_breakdown_model.pkl
requirements.txt
README.md
HERMES_AGENT_ARCHITECTURE.md
INDUSTRIAL_DEPLOYMENT_GUIDE.md
PROJECT_FILE_GUIDE.md
```

Generated folders/files:

```text
results/
__pycache__/
```

`results/` is created when reports or legacy prediction outputs are generated. `__pycache__/` is Python runtime cache and should not be treated as source code.

## 6. File-By-File Guide

### `app.py`

Main application entry point.

Current responsibility:

```python
from backend.hermes_api import run
```

Running `python app.py` starts the Flask server and the Hermes Agent dashboard.

### `backend/hermes_api.py`

The Flask backend API.

Responsibilities:

- serve the frontend files
- expose live Hermes Agent state
- accept real telemetry JSON
- generate reports
- generate simulator datasets
- retrain/evaluate the model
- expose asset configuration
- expose credibility/production-readiness metadata

Important endpoints:

```text
GET  /
GET  /api/state
GET  /api/assets
GET  /api/credibility
POST /api/telemetry
POST /api/simulator/generate
POST /api/model/retrain
POST /api/model/evaluate
POST /api/report
GET  /reports/<filename>
```

### `agent/hermes_agent.py`

The central agent orchestration layer.

Hermes Agent responsibilities:

- observe telemetry
- call RUL prediction
- call anomaly detection
- calculate failure probability
- generate failure explanation
- recommend maintenance
- simulate future degradation
- create alerts
- create reports during critical states
- maintain live logs and thought process

Important methods:

- `tick()` - runs one full monitoring cycle
- `ingest_telemetry(payload)` - accepts external plant telemetry
- `reload_stream(source_path)` - reloads the active CSV stream
- `report()` - generates a maintenance PDF

### `tools/industrial_tools.py`

Reusable JSON-returning tool functions.

These are the agent's callable tools:

- `predict_rul(sensor_window, engine)`
- `detect_anomaly(sensor_data)`
- `calculate_failure_probability(rul_hours, anomaly_score, sensor_data)`
- `generate_failure_analysis(sensor_data, rul_hours, anomaly, failure_probability)`
- `recommend_maintenance(rul_hours, anomaly_score, failure_probability)`
- `simulate_future_degradation(sensor_data, rul_hours, failure_probability)`
- `send_alert(level, message)`
- `generate_report(machine_state)`

Every tool returns structured JSON-like dictionaries so it can be reused by APIs, dashboards, and future agent frameworks.

### `ml/prediction_engine.py`

Runtime model wrapper.

Responsibilities:

- load `machine_breakdown_model.pkl`
- train a fallback model if the pickle is missing or incompatible
- build features using the legacy pipeline's feature builder
- predict RUL from the current telemetry window
- return health percentage and confidence

The model output is normalized back to hours using the saved training scale.

### `ml/model_ops.py`

Model operations utilities.

Responsibilities:

- `retrain_rul_model()` retrains the Ridge model from a labeled CSV
- `evaluate_model()` evaluates the saved model against a labeled test CSV

This is used by:

```text
POST /api/model/retrain
POST /api/model/evaluate
```

### `maintenance_pipeline.py`

Legacy but still useful ML training/prediction pipeline.

Responsibilities:

- parse command-line prediction arguments
- train the Ridge Regression model
- build rolling features
- save model bundle
- calculate validation metrics
- create prediction CSV/plot if used directly

It is still important because `ml/prediction_engine.py` and `ml/model_ops.py` reuse its robust functions:

- `train_model()`
- `build_features()`
- `optional_target_column()`
- `regression_metrics()`

### `data_simulator.py`

Dataset generator.

It creates realistic sample machine degradation datasets.

Commands:

```powershell
python data_simulator.py few-days
python data_simulator.py few-days --profile warning
python data_simulator.py few-days --profile critical
python data_simulator.py healthy
python data_simulator.py long-life
```

Profiles:

- `healthy` - high remaining life
- `watch` - medium degradation
- `warning` - meaningful risk
- `critical` - near-failure data
- `auto` - random profile

Yes, data generated here can be used to predict breakdown. It is best for demos, testing, and model rehearsal. It is not proof of real-world accuracy unless validated against real machine history.

### `simulation/sensor_stream.py`

Telemetry stream adapter.

Responsibilities:

- replay CSV rows as live telemetry
- maintain rolling history window
- accept external telemetry through an in-memory queue
- canonicalize incoming data using tag mappings
- reload CSV datasets when simulator data is generated

Data source priority:

1. If external telemetry is queued, use it first.
2. Otherwise replay the configured CSV file.

### `simulation/future_degradation.py`

Future failure simulator.

It projects:

- current machine state
- 24 hours later
- 48 hours later
- 72 hours later
- estimated failure point

Outputs include:

- health percent
- vibration
- temperature
- bearing wear
- lubricant quality
- failure probability

### `simulation/simulator_ops.py`

API helper around `data_simulator.py`.

Used by:

```text
POST /api/simulator/generate
```

It generates a new dataset, updates `datasets/few_days_machine_dataset.csv`, and reloads the live stream.

### `ingestion/schema.py`

Canonical telemetry schema.

This file makes real machine integration easier.

It maps different field names into the names expected by the ML model:

- `vibration_rms_mm_s`
- `temperature_c`
- `rpm`
- `acoustic_noise_db`
- `bearing_wear_percent`
- `lubricant_quality_percent`
- `remaining_useful_life_hours`

It also calculates data quality:

- missing required fields
- missing optional fields
- prediction readiness
- completeness score

### `ingestion/asset_config.py`

Loads the machine asset configuration from:

```text
config/machine_assets.json
```

### `config/machine_assets.json`

Machine configuration file.

Defines:

- active asset id
- display name
- machine type
- plant area
- telemetry source
- CSV replay file
- sampling interval
- plant tag mapping
- alert thresholds

Use this file when connecting real machines with different tag names.

Example mapping:

```json
"tag_map": {
  "vibration_rms_mm_s": "PLC1.DB20.VIB_RMS",
  "temperature_c": "PLC1.DB20.BRG_TEMP",
  "rpm": "PLC1.DB20.MOTOR_RPM"
}
```

### `frontend/index.html`

Dashboard structure.

Contains:

- top navigation
- sidebar
- Dashboard view
- Live Monitoring view
- Predictions view
- Analytics view
- Reports view
- Alerts view
- Settings view

### `frontend/styles.css`

Enterprise dark-theme styling.

Controls:

- layout
- responsive behavior
- dashboard cards
- tables
- charts
- navigation
- mobile sidebar
- settings forms
- alert colors

### `frontend/app.js`

Frontend application logic.

Responsibilities:

- poll `/api/state`
- switch sidebar views
- draw charts on canvas
- render sensor tables
- render risk matrices
- render alerts/logs/decisions
- generate reports
- generate simulator datasets
- evaluate model
- control refresh interval, trend window, density, pause/resume

Important frontend views:

- `dashboard`
- `monitoring`
- `predictions`
- `analytics`
- `reports`
- `alerts`
- `settings`

### `reports/maintenance_report.py`

PDF report generator.

Creates:

```text
results/last_200_hours_maintenance_report.pdf
```

Report includes:

- alert level
- machine health
- RUL
- failure probability
- anomaly severity
- failure explanation
- maintenance recommendation
- future degradation chart

### `project_paths.py`

Shared path constants.

Defines:

- project root
- datasets folder
- results folder
- model path
- default training dataset
- default prediction dataset

### `machine_breakdown_model.pkl`

Saved trained model bundle.

Contains:

- Ridge Regression pipeline
- feature column list
- sensor column list
- target column
- target unit
- target max scale
- validation metrics

If deleted, the application can retrain from the default training dataset, but startup will take longer.

### `requirements.txt`

Python dependencies:

- Flask
- pandas
- numpy
- matplotlib
- scipy
- joblib
- scikit-learn

### `README.md`

Short project overview and quick commands.

### `HERMES_AGENT_ARCHITECTURE.md`

Architecture summary for reviewers.

### `INDUSTRIAL_DEPLOYMENT_GUIDE.md`

Deployment guide explaining:

- CSV replay vs real telemetry
- telemetry API contract
- tag mapping
- simulator usage
- production-readiness limitations
- real-machine integration path

## 7. Dataset Files

Important dataset files:

```text
datasets/machine_vibration_training_dataset.csv
datasets/machine_vibration_testing_dataset.csv
datasets/few_days_machine_dataset.csv
datasets/extended_training_dataset.csv
datasets/long_life_machine_dataset.csv
```

`machine_vibration_training_dataset.csv`

Primary training dataset for the default Ridge RUL model.

`few_days_machine_dataset.csv`

Default live replay dataset used by Hermes dashboard.

`extended_training_dataset.csv`

Long-life training data generated by simulator.

`long_life_machine_dataset.csv`

Long-life test/replay data.

When the simulator is run, it may create numbered history files such as:

```text
few_days_machine_dataset_1.csv
few_days_machine_dataset_7.csv
```

These are generated datasets, not required source files. They are useful only when you intentionally want to compare multiple simulated scenarios.

## 8. How To Use Simulator Data

From command line:

```powershell
python data_simulator.py few-days --profile warning --seed 42
```

From dashboard:

1. Open Settings.
2. Choose Simulator Profile.
3. Enter seed.
4. Click `Generate and Load Simulator Data`.

This updates:

```text
datasets/few_days_machine_dataset.csv
```

Then Hermes reloads the stream and starts predicting against the generated data.

## 9. How To Send Real Machine Data

Send JSON to:

```http
POST http://127.0.0.1:5000/api/telemetry
```

Example PowerShell command:

```powershell
$body = @{
  vibration_rms_mm_s = 4.7
  temperature_c = 68.2
  rpm = 1450
  acoustic_noise_db = 72.4
  bearing_wear_percent = 58
  lubricant_quality_percent = 43
} | ConvertTo-Json

Invoke-RestMethod -Uri "http://127.0.0.1:5000/api/telemetry" -Method Post -Body $body -ContentType "application/json"
```

Required fields:

- `vibration_rms_mm_s`
- `temperature_c`
- `rpm`

Optional but valuable fields:

- `acoustic_noise_db`
- `bearing_wear_percent`
- `lubricant_quality_percent`

If a real plant uses different names, configure `config/machine_assets.json`.

## 10. How Prediction Works

The model is a Ridge Regression pipeline:

```python
make_pipeline(StandardScaler(), Ridge(alpha=1.0))
```

The feature builder creates rolling features for each sensor:

- current value
- difference from previous value
- exponential moving average
- rolling mean
- rolling standard deviation
- rolling minimum
- rolling maximum

Rolling windows:

```text
5, 15, 30
```

This helps the model understand both current state and recent trend.

The model predicts normalized RUL. The system converts it back to hours using the training scale stored in `machine_breakdown_model.pkl`.

## 11. How Hermes Decides Maintenance

Hermes uses:

- predicted RUL
- anomaly score
- failure probability
- vibration
- temperature
- bearing wear
- lubricant quality

Decision levels:

```text
GREEN  -> Continue monitoring
YELLOW -> Schedule inspection within next shift
ORANGE -> Replace bearing and reduce operational load
RED    -> Emergency shutdown required
```

This logic lives in:

```text
tools/industrial_tools.py
```

## 12. How Reports Work

Dashboard:

1. Open Reports.
2. Click Generate Maintenance PDF.

API:

```http
POST /api/report
```

Output:

```text
results/last_200_hours_maintenance_report.pdf
```

The PDF is generated dynamically. It is not stored as source code.

## 13. How To Evaluate Model Credibility

Dashboard:

1. Open Settings.
2. Click Evaluate Current Model.

API:

```http
POST /api/model/evaluate
```

Example payload:

```json
{
  "test_csv": "datasets/few_days_machine_dataset.csv"
}
```

Metrics returned:

- MAE
- RMSE
- R2
- sample count

Important: metrics are only production-credible when the test CSV is real unseen machine history, not synthetic simulator data.

## 14. How To Retrain The Model

API:

```http
POST /api/model/retrain
```

Example payload:

```json
{
  "train_csv": "datasets/machine_vibration_training_dataset.csv"
}
```

The training CSV must include a RUL target column such as:

- `remaining_useful_life_hours`
- `actual_remaining_life`
- `rul`
- `remaining_life`

## 15. Production Deployment Notes

This is currently:

```text
Industrial AI prototype with real-ingestion pathway.
```

It is not yet:

```text
Certified production control software.
```

To make it real-machine ready:

1. Connect actual telemetry source: MQTT, OPC UA, Modbus TCP, Kafka, SCADA historian, Azure IoT Hub, AWS IoT SiteWise.
2. Map plant tags in `config/machine_assets.json`.
3. Collect real historical machine data.
4. Add maintenance events and confirmed failure labels.
5. Retrain per asset type.
6. Evaluate on unseen real timelines.
7. Track false positives and false negatives.
8. Add technician feedback.
9. Keep Hermes recommendation-only until safety review approves automation.

## 16. Removed Obsolete Files

The following old/generated files were removed because they no longer belong to the current Hermes platform:

- `dashboard.py` - old Streamlit dashboard replaced by Flask + `frontend/`
- `project_templates.py` - old dashboard template generator
- `debug_predictions.py` - old ad hoc debug script
- `WORKING_DOCUMENTATION.md` - outdated Streamlit documentation
- `results.json` - old generated dashboard summary
- generated files under `results/`
- `__pycache__/` folders

## 17. Normal Development Commands

Install:

```powershell
pip install -r requirements.txt
```

Run:

```powershell
python app.py
```

Generate simulator data:

```powershell
python data_simulator.py few-days --profile critical --seed 42
```

Compile-check Python:

```powershell
python -m compileall agent backend ingestion ml reports simulation tools app.py
```

Check frontend JavaScript:

```powershell
node --check frontend\app.js
```

## 18. Final Mental Model

Think of the project in four layers:

```text
Ingestion Layer
  config/, ingestion/, simulation/

Intelligence Layer
  ml/, tools/, agent/

Service Layer
  backend/, reports/

Experience Layer
  frontend/
```

The most important file is:

```text
agent/hermes_agent.py
```

because it ties together telemetry, prediction, tools, reasoning, decisions, alerts, reports, and dashboard state.
