from __future__ import annotations

from pathlib import Path
from typing import Any

from data_simulator import generate_few_days_dataset, save_latest_and_numbered


def generate_simulated_machine_data(seed: int = 42, profile: str = "warning") -> dict[str, Any]:
    df, selected_profile = generate_few_days_dataset(seed=seed, profile=profile)
    dataset_number, latest_path, numbered_path = save_latest_and_numbered(
        df,
        "few_days_machine_dataset.csv",
        "few_days_machine_dataset",
    )
    return {
        "dataset_number": dataset_number,
        "profile": selected_profile,
        "latest_path": str(latest_path),
        "numbered_path": str(numbered_path),
        "samples": len(df),
        "rul_min_hours": float(df["remaining_useful_life_hours"].min()),
        "rul_max_hours": float(df["remaining_useful_life_hours"].max()),
    }
