from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from project_paths import RESULTS_DIR, ensure_project_dirs


def generate_maintenance_pdf(machine_state: dict[str, Any]) -> Path:
    ensure_project_dirs()
    report_path = RESULTS_DIR / "last_200_hours_maintenance_report.pdf"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction = machine_state.get("prediction", {})
    analysis = machine_state.get("analysis", {})
    decision = machine_state.get("decision", {})
    anomaly = machine_state.get("anomaly", {})
    future = machine_state.get("future", {}).get("timeline", [])

    with PdfPages(report_path) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.patch.set_facecolor("#081019")
        ax = fig.add_subplot(111)
        ax.axis("off")
        lines = [
            "LAST 200 HOURS",
            "Autonomous Hermes Agent Maintenance Report",
            "",
            f"Generated: {timestamp}",
            f"Alert Level: {machine_state.get('alert_level', 'UNKNOWN')}",
            f"Machine Health: {prediction.get('health_percent', 0)}%",
            f"Remaining Useful Life: {prediction.get('rul_hours', 0)} hours",
            f"Failure Probability: {machine_state.get('failure_probability', 0)}",
            f"Anomaly Severity: {anomaly.get('severity', 'unknown')}",
            "",
            "Failure Explanation:",
            analysis.get("summary", "No analysis available."),
            "",
            "Maintenance Recommendation:",
            decision.get("action", "No action available."),
            decision.get("rationale", ""),
        ]
        ax.text(0.07, 0.95, "\n".join(lines), va="top", color="#dff7ff", fontsize=12, linespacing=1.55)
        pdf.savefig(fig, facecolor=fig.get_facecolor())
        plt.close(fig)

        if future:
            fig, ax = plt.subplots(figsize=(11, 6))
            hours = [point["hours_from_now"] for point in future]
            health = [point["health_percent"] for point in future]
            risk = [point["failure_probability"] * 100 for point in future]
            ax.plot(hours, health, color="#45f3ff", linewidth=3, label="Health %")
            ax.plot(hours, risk, color="#ff3f6e", linewidth=3, label="Failure Risk %")
            ax.set_title("Future Degradation Timeline")
            ax.set_xlabel("Hours from now")
            ax.set_ylabel("Percent")
            ax.grid(alpha=0.25)
            ax.legend()
            pdf.savefig(fig)
            plt.close(fig)

    return report_path
