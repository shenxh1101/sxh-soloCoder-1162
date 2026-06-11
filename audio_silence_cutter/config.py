import json
import os
from typing import Any, Dict, Optional


DEFAULT_CONFIG: Dict[str, Any] = {
    "threshold_db": -40.0,
    "min_silence_ms": 500.0,
    "buffer_before_ms": 200.0,
    "buffer_after_ms": 200.0,
    "output_dir": "./output",
    "output_format": None,
    "smart_merge": False,
    "min_segment_ms": 1000.0,
    "export_envelope": False,
    "save_report": False,
    "no_report": False,
    "batch_summary": False,
}

CONFIG_KEYS = set(DEFAULT_CONFIG.keys())


def load_config(config_path: str) -> Dict[str, Any]:
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    filtered = {}
    for key, value in config.items():
        if key in CONFIG_KEYS:
            filtered[key] = value

    return filtered


def save_config_template(path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_CONFIG, f, indent=2, ensure_ascii=False)

    return path


def merge_config(
    cli_args: Dict[str, Any],
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)

    if config_path:
        file_config = load_config(config_path)
        merged.update(file_config)

    for key in CONFIG_KEYS:
        cli_value = cli_args.get(key)
        if cli_value is not None:
            merged[key] = cli_value

    return merged