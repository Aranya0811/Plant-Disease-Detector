"""Load project configuration from config.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT_DIR / "config.yaml"


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or CONFIG_PATH
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for key in ("data_dir", "model_save_path", "class_map_path", "logs_dir"):
        if key in config:
            config[key] = str(ROOT_DIR / config[key])

    return config
