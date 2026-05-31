# LAST 200 HOURS - Hermes Agent Architecture

Tagline: "Every machine whispers before it dies. Hermes Agent learns to listen."

## Runtime Flow

```text
CSV Replay / POST /api/telemetry / Plant Adapter
  -> Canonical Telemetry Schema
  -> Sensor Stream
  -> Prediction Engine
  -> Hermes Agent
       -> tool: detect_anomaly
       -> tool: predict_rul
       -> tool: calculate_failure_probability
       -> tool: generate_failure_analysis
       -> tool: recommend_maintenance
       -> tool: simulate_future_degradation
       -> tool: send_alert
       -> tool: generate_report
  -> Flask API
  -> Cinematic dashboard
```

## Folder Ownership

- `agent/` - Hermes Agent orchestration loop and autonomous reasoning state.
- `tools/` - JSON-returning callable tools used by Hermes.
- `ml/` - model loading, training fallback, and RUL prediction wrapper.
- `simulation/` - live CSV telemetry stream and future degradation projection.
- `reports/` - automated PDF maintenance report generation.
- `backend/` - Flask API that exposes agent state and report downloads.
- `ingestion/` - real telemetry schema normalization and asset configuration helpers.
- `frontend/` - full-screen LAST 200 HOURS dashboard experience.
- `datasets/` - existing machine telemetry and RUL datasets.
- `results/` - generated reports, plots, and prediction outputs.

## API Surface

- `GET /` - opens the dashboard.
- `GET /api/state` - advances Hermes Agent one monitoring cycle and returns the full machine state.
- `POST /api/telemetry` - accepts real plant telemetry JSON and queues it for Hermes.
- `GET /api/assets` - returns configured industrial assets and tag maps.
- `POST /api/simulator/generate` - generates a new simulator dataset and reloads the stream.
- `POST /api/model/retrain` - retrains the RUL model from a labeled CSV.
- `POST /api/model/evaluate` - evaluates the saved model against a labeled CSV.
- `GET /api/credibility` - returns production-readiness notes and next steps.
- `POST /api/report` - generates the latest maintenance PDF.
- `GET /reports/<filename>` - downloads generated reports.

## Agent Decisions

Hermes autonomously chooses one of four maintenance states:

- `GREEN` - Continue monitoring.
- `YELLOW` - Schedule inspection within next shift.
- `ORANGE` - Replace bearing and reduce operational load.
- `RED` - Emergency shutdown required.

The decision is based on RUL, anomaly score, failure probability, vibration trend, thermal load, bearing wear, and lubricant quality.

## Demo Command

```powershell
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```
