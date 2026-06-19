from __future__ import annotations

from pathlib import Path

import yaml

from equity_transformer.gui.artifacts import read_json_if_exists
from equity_transformer.gui.configs import read_config, write_config_text


def apply_selected_factor_to_strategy(
    root: str | Path = ".",
    factor_index: int = 0,
) -> dict[str, object]:
    base = Path(root)
    manifest = read_json_if_exists(base / "artifacts/factors/selected_factors.json")
    selected = manifest.get("selected_factors", [])
    if not selected:
        raise ValueError("No selected factors are available.")
    if factor_index < 0 or factor_index >= len(selected):
        raise IndexError(f"factor_index out of range: {factor_index}")

    strategy_config = read_config("strategy", base)
    strategy_config["score_column"] = selected[factor_index]
    text = yaml.safe_dump(strategy_config, sort_keys=False)
    write_config_text("strategy", text, base)
    return strategy_config
