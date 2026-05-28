from __future__ import annotations

from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from ml.prediction_engine import PredictionEngine
from ingestion.schema import telemetry_quality
from simulation.sensor_stream import SensorStream
from tools.industrial_tools import (
    calculate_failure_probability,
    detect_anomaly,
    generate_failure_analysis,
    generate_report,
    predict_rul,
    recommend_maintenance,
    send_alert,
    simulate_future_degradation,
)


class HermesAgent:
    """Autonomous reliability engineer for the LAST 200 HOURS system."""

    def __init__(self, stream_path: Path | None = None) -> None:
        self.engine = PredictionEngine()
        self.stream = SensorStream(stream_path)
        self.logs: deque[dict[str, Any]] = deque(maxlen=80)
        self.alerts: deque[dict[str, Any]] = deque(maxlen=30)
        self.last_state: dict[str, Any] | None = None
        self.asset = self.stream.asset

    def _log(self, message: str, level: str = "INFO") -> dict[str, Any]:
        item = {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "message": message,
        }
        self.logs.appendleft(item)
        return item

    def _last_words(self, alert_level: str, rul_hours: float, anomaly_score: float) -> list[str]:
        if alert_level == "RED":
            return [
                "Operational confidence dropping below safe threshold.",
                f"I can continue for approximately {max(rul_hours, 0):.0f} more hours.",
                "Thermal imbalance detected. Critical failure approaching.",
            ]
        if alert_level == "ORANGE":
            return [
                "Bearing instability escalating rapidly.",
                f"My remaining useful life is compressing toward {rul_hours:.0f} hours.",
            ]
        if alert_level == "YELLOW" or anomaly_score >= 0.42:
            return ["Harmonic vibration has changed. I am asking to be inspected."]
        return ["All core signals remain coherent. I am still operating with confidence."]

    def tick(self) -> dict[str, Any]:
        sample = self.stream.next_sample()
        quality = telemetry_quality(sample.data)
        self._log("[Hermes] Monitoring vibration stream...")

        anomaly_tool = detect_anomaly(sample.data)
        anomaly = anomaly_tool["result"]
        self._log(f"[Hermes] Anomaly severity classified as {anomaly['severity']}.")

        prediction_tool = predict_rul(sample.window, self.engine)
        prediction = prediction_tool["result"]
        self._log(f"[Hermes] Estimated machine life: {prediction['rul_hours']:.1f} hours.")

        probability_tool = calculate_failure_probability(
            prediction["rul_hours"],
            anomaly["anomaly_score"],
            sample.data,
        )
        failure_probability = probability_tool["result"]["failure_probability"]
        self._log(f"[Hermes] Failure probability rising to {failure_probability:.0%}.")

        analysis_tool = generate_failure_analysis(sample.data, prediction["rul_hours"], anomaly_tool, failure_probability)
        analysis = analysis_tool["result"]
        self._log(f"[Hermes] {analysis['primary_failure_mode'].title()} pattern evaluated.")

        decision_tool = recommend_maintenance(prediction["rul_hours"], anomaly["anomaly_score"], failure_probability)
        decision = decision_tool["result"]
        self._log(f"[Hermes] Maintenance decision: {decision['action']}.", decision["priority"])

        future_tool = simulate_future_degradation(sample.data, prediction["rul_hours"], failure_probability)
        future = future_tool["result"]

        alert = send_alert(decision["priority"], decision["action"])["result"]
        if decision["priority"] != "GREEN":
            self.alerts.appendleft(alert)

        thought_process = [
            "[Hermes] Comparing live waveform against trained degradation memory.",
            f"[Hermes] Health field stabilized at {prediction['health_percent']:.1f}%.",
            f"[Hermes] Risk model reports {failure_probability:.0%} failure probability.",
            f"[Hermes] Evidence: {analysis['summary']}",
            f"[Hermes] Autonomous action: {decision['action']}.",
        ]

        state = {
            "project": "LAST 200 HOURS",
            "tagline": "Every machine whispers before it dies. Hermes Agent learns to listen.",
            "sample_index": sample.index,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "sensor_data": sample.data,
            "asset": self.asset,
            "data_source": {
                "mode": self.stream.source_mode,
                "csv_path": str(self.stream.source_path),
                "external_buffer": len(self.stream.external_samples),
            },
            "data_quality": quality,
            "prediction": prediction,
            "anomaly": anomaly,
            "failure_probability": failure_probability,
            "analysis": analysis,
            "decision": decision,
            "future": future,
            "alert_level": decision["priority"],
            "alert": alert,
            "alerts": list(self.alerts),
            "logs": list(self.logs),
            "thought_process": thought_process,
            "machine_last_words": self._last_words(decision["priority"], prediction["rul_hours"], anomaly["anomaly_score"]),
        }

        if decision["priority"] == "RED":
            report = generate_report(state)["result"]
            state["report"] = report

        self.last_state = state
        return state

    def ingest_telemetry(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample = self.stream.push_sample(payload)
        quality = telemetry_quality(sample)
        self._log("[Hermes] External plant telemetry accepted.", "INFO")
        return {"accepted": quality["ready_for_prediction"], "sample": sample, "data_quality": quality}

    def reload_stream(self, source_path: Path) -> dict[str, Any]:
        self.stream.reload_csv(source_path)
        self._log(f"[Hermes] CSV telemetry source changed to {source_path.name}.", "INFO")
        return {"csv_path": str(source_path), "samples": len(self.stream.frame)}

    def report(self) -> dict[str, Any]:
        if self.last_state is None:
            self.tick()
        assert self.last_state is not None
        return generate_report(self.last_state)["result"]
