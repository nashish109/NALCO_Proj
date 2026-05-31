# LAST 200 HOURS - Industrial Deployment Guide

This project now supports two telemetry modes:

1. CSV replay for demos and simulator-generated degradation scenarios.
2. External telemetry ingestion through `POST /api/telemetry`.

## Where The Data Comes From Today

By default, Hermes reads:

```text
datasets/few_days_machine_dataset.csv
```

The stream is configured in:

```text
config/machine_assets.json
```

Every dashboard refresh advances the stream by one row. This is useful for demos, validation rehearsals, and generated degradation scenarios, but it is not the same as live PLC/SCADA data.

## Can Simulator Data Predict Breakdown?

Yes, simulator-generated data can be used to test breakdown prediction when:

- The simulator emits the same sensor columns used by the model.
- RUL labels exist, such as `remaining_useful_life_hours`.
- You retrain or evaluate the model against the simulator dataset.

Use it for:

- Demonstrations.
- Stress testing alert states.
- Checking UI and agent behavior.
- Teaching the pipeline what a synthetic degradation curve looks like.

Do not treat simulator-only performance as production proof. Real credibility requires real historical machine data and actual maintenance/failure records.

## Real Telemetry Contract

Send live machine readings to:

```http
POST /api/telemetry
Content-Type: application/json
```

Example:

```json
{
  "timestamp": "2026-05-28T14:22:00Z",
  "vibration_rms_mm_s": 4.7,
  "temperature_c": 68.2,
  "rpm": 1450,
  "acoustic_noise_db": 72.4,
  "bearing_wear_percent": 58.0,
  "lubricant_quality_percent": 43.0
}
```

Minimum recommended fields:

- `vibration_rms_mm_s`
- `temperature_c`
- `rpm`

Optional fields improve diagnosis:

- `acoustic_noise_db`
- `bearing_wear_percent`
- `lubricant_quality_percent`

## Plant Tag Mapping

If real tags use different names, map them in:

```text
config/machine_assets.json
```

Example:

```json
"tag_map": {
  "vibration_rms_mm_s": "PLC1.DB20.VIB_RMS",
  "temperature_c": "PLC1.DB20.BRG_TEMP",
  "rpm": "PLC1.DB20.MOTOR_RPM"
}
```

Then post payloads using the plant tag names.

## Recommended Production Path

1. Connect real telemetry using MQTT, OPC UA, Modbus, Kafka, Azure IoT Hub, or historian export.
2. Map raw plant tags into canonical model fields.
3. Collect historical machine records with maintenance dates and confirmed failures.
4. Retrain the RUL model using real asset history.
5. Evaluate on unseen real machine timelines.
6. Track false positives, false negatives, alert lead time, and maintenance engineer feedback.
7. Keep Hermes recommendation-only until safety engineers approve automated control actions.

## Useful API Endpoints

- `GET /api/state` - advance one monitoring cycle and return Hermes state.
- `POST /api/telemetry` - inject live external telemetry.
- `POST /api/simulator/generate` - generate and load simulator data.
- `POST /api/model/retrain` - retrain RUL model from a CSV.
- `POST /api/model/evaluate` - evaluate model on a labeled CSV.
- `GET /api/credibility` - show production-readiness notes.

## Credibility

Current status:

```text
Industrial AI prototype, not certified production control software.
```

What is strong:

- Modular ingestion architecture.
- Agent orchestration.
- RUL prediction pipeline.
- Anomaly/risk scoring.
- Alerting and report generation.
- Enterprise dashboard.

What still needs real-world proof:

- Model trained on real failures.
- Validated alert thresholds.
- Asset-specific baseline behavior.
- Maintenance-team feedback loop.
- Safety review for any control-system action.
