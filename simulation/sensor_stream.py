from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import deque

import pandas as pd

from ingestion.asset_config import active_asset
from ingestion.schema import canonicalize_telemetry
from project_paths import DATASETS_DIR


@dataclass
class StreamSample:
    index: int
    data: dict[str, Any]
    window: pd.DataFrame


class SensorStream:
    """Loops through a CSV as if it were a live machine telemetry feed."""

    def __init__(
        self,
        source_path: Path | None = None,
        window_size: int = 48,
    ) -> None:
        self.asset = active_asset()
        configured_path = Path(str(self.asset.get("csv_path", DATASETS_DIR / "few_days_machine_dataset.csv")))
        if not configured_path.is_absolute():
            configured_path = Path(__file__).resolve().parents[1] / configured_path
        self.source_path = source_path or configured_path
        self.window_size = window_size
        self.frame = pd.read_csv(self.source_path)
        self.pointer = 0
        self.external_samples: deque[dict[str, Any]] = deque(maxlen=500)
        self.history: deque[dict[str, Any]] = deque(maxlen=window_size)
        self.source_mode = str(self.asset.get("telemetry_source", "csv_replay"))

    def push_sample(self, payload: dict[str, Any]) -> dict[str, Any]:
        sample = canonicalize_telemetry(payload, self.asset.get("tag_map", {}))
        self.external_samples.append(sample)
        return sample

    def next_sample(self) -> StreamSample:
        if self.external_samples:
            row = self.external_samples.popleft()
            self.history.append(row)
            window = pd.DataFrame(list(self.history))
            return StreamSample(index=self.pointer, data=row, window=window)

        if self.pointer >= len(self.frame):
            self.pointer = 0

        row = canonicalize_telemetry(
            self.frame.iloc[self.pointer].to_dict(),
            self.asset.get("tag_map", {}),
        )
        self.history.append(row)
        start = max(0, self.pointer - self.window_size + 1)
        window = self.frame.iloc[start : self.pointer + 1].copy()
        sample = StreamSample(index=self.pointer, data=row, window=window)
        self.pointer += 1
        return sample

    def reload_csv(self, source_path: Path) -> None:
        self.source_path = source_path
        self.frame = pd.read_csv(self.source_path)
        self.pointer = 0
        self.history.clear()
