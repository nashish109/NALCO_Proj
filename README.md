````markdown
# LAST 200 HOURS

An AI-powered predictive maintenance system built with Hermes Agent.

LAST 200 HOURS analyzes industrial telemetry data to predict equipment failures, estimate Remaining Useful Life (RUL), detect anomalies, generate maintenance recommendations, and learn from historical machine behavior.

Built for the Hermes Agent Challenge.

---

## What It Does

- Predicts Remaining Useful Life (RUL)
- Detects abnormal machine behavior
- Calculates failure probability
- Generates maintenance recommendations
- Simulates future equipment degradation
- Produces PDF maintenance reports
- Ingests real-time telemetry data
- Evaluates and retrains predictive models

---

## Hermes Agent

Hermes Agent serves as the reasoning and decision-making layer of the platform.

It:

- Analyzes telemetry insights
- Investigates anomalies
- Explains failure risks
- Generates maintenance recommendations
- Produces maintenance reports
- Supports operational decision-making

Unlike traditional predictive maintenance systems that stop at predictions, Hermes Agent helps transform predictions into actionable maintenance intelligence.

---

## Architecture

```text
Telemetry Data
      │
      ▼
Machine Learning Models
      │
      ├── RUL Prediction
      ├── Failure Scoring
      └── Anomaly Detection
      │
      ▼
   Hermes Agent
      │
      ▼
Recommendations & Reports
````

---

## Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

## Documentation

* PROJECT_FILE_GUIDE.md
* HERMES_AGENT_ARCHITECTURE.md
* INDUSTRIAL_DEPLOYMENT_GUIDE.md

---

## Future Roadmap

* Real-Time IoT Integration
* Multi-Agent Collaboration
* Autonomous Maintenance Planning
* Digital Twin Integration

---

"Don't just predict failures. Learn from them."

```
```
