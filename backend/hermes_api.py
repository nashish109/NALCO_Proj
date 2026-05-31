from __future__ import annotations

from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory

from agent.hermes_agent import HermesAgent
from ingestion.asset_config import load_asset_config
from ml.model_ops import evaluate_model, retrain_rul_model
from project_paths import RESULTS_DIR
from simulation.simulator_ops import generate_simulated_machine_data


ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIR = ROOT / "frontend"


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
    app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
    agent = HermesAgent()

    @app.after_request
    def no_cache(response):
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    @app.get("/")
    def index():
        return send_from_directory(FRONTEND_DIR, "index.html")

    @app.get("/api/state")
    def state():
        return jsonify(agent.tick())

    @app.get("/api/assets")
    def assets():
        return jsonify(load_asset_config())

    @app.post("/api/telemetry")
    def telemetry():
        payload = request.get_json(force=True, silent=False)
        if not isinstance(payload, dict):
            return jsonify({"error": "Expected a JSON object telemetry payload."}), 400
        return jsonify(agent.ingest_telemetry(payload))

    @app.post("/api/simulator/generate")
    def simulator_generate():
        payload = request.get_json(silent=True) or {}
        seed = int(payload.get("seed", 42))
        profile = str(payload.get("profile", "warning"))
        result = generate_simulated_machine_data(seed=seed, profile=profile)
        agent.reload_stream(Path(result["latest_path"]))
        return jsonify(result)

    @app.post("/api/model/retrain")
    def model_retrain():
        payload = request.get_json(silent=True) or {}
        train_csv = Path(str(payload.get("train_csv", "datasets/machine_vibration_training_dataset.csv")))
        if not train_csv.is_absolute():
            train_csv = ROOT / train_csv
        return jsonify(retrain_rul_model(train_csv))

    @app.post("/api/model/evaluate")
    def model_evaluate():
        payload = request.get_json(silent=True) or {}
        test_csv = Path(str(payload.get("test_csv", "datasets/few_days_machine_dataset.csv")))
        if not test_csv.is_absolute():
            test_csv = ROOT / test_csv
        return jsonify(evaluate_model(test_csv))

    @app.get("/api/credibility")
    def credibility():
        return jsonify(
            {
                "current_mode": agent.stream.source_mode,
                "credibility_level": "industrial prototype",
                "production_readiness": {
                    "architecture": "ready for plant adapters",
                    "data_source": "CSV replay by default; external telemetry supported through POST /api/telemetry",
                    "model": "must be validated against real historical failure and maintenance records before operational use",
                    "safety": "recommendation-only; do not connect shutdown commands without certified safety review",
                },
                "real_machine_next_steps": [
                    "Connect OPC UA, MQTT, Modbus, historian, or IoT Hub data into POST /api/telemetry.",
                    "Map plant tags in config/machine_assets.json.",
                    "Retrain using real asset history and failure labels.",
                    "Evaluate on unseen real machine histories with /api/model/evaluate.",
                    "Add technician feedback labels for false positives and confirmed faults.",
                ],
            }
        )

    @app.post("/api/report")
    def report():
        return jsonify(agent.report())

    @app.get("/reports/<path:filename>")
    def reports(filename: str):
        return send_from_directory(RESULTS_DIR, filename, as_attachment=True)

    return app


def run() -> None:
    create_app().run(host="127.0.0.1", port=5000, debug=False)
