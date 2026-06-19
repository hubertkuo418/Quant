from __future__ import annotations

import json
from pathlib import Path

import pytest

from equity_transformer.gui.configs import read_config
from equity_transformer.gui.strategy_builder import apply_selected_factor_to_strategy


def test_apply_selected_factor_to_strategy_updates_score_column(tmp_path: Path) -> None:
    factor_dir = tmp_path / "artifacts" / "factors"
    config_dir = tmp_path / "configs"
    factor_dir.mkdir(parents=True)
    config_dir.mkdir()
    (factor_dir / "selected_factors.json").write_text(
        json.dumps({"selected_factors": ["alpha101_001", "return_20d"]}),
        encoding="utf-8",
    )
    (config_dir / "strategy.yaml").write_text(
        "score_column: old\nstrategy_type: long_only_top_k\n",
        encoding="utf-8",
    )

    updated = apply_selected_factor_to_strategy(tmp_path, factor_index=1)

    assert updated["score_column"] == "return_20d"
    assert read_config("strategy", tmp_path)["score_column"] == "return_20d"


def test_apply_selected_factor_to_strategy_requires_selected_factors(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="No selected factors"):
        apply_selected_factor_to_strategy(tmp_path)
