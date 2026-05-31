from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
ASSET_CONFIG_PATH = ROOT / "config" / "machine_assets.json"


def load_asset_config(path: Path = ASSET_CONFIG_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def active_asset(path: Path = ASSET_CONFIG_PATH) -> dict[str, Any]:
    config = load_asset_config(path)
    active_id = config["active_asset_id"]
    return config["assets"][active_id]
