from __future__ import annotations

from pathlib import Path

import pytest

from equity_transformer.gui.configs import (
    list_config_entries,
    read_config,
    read_config_text,
    write_config_text,
)


def test_config_catalog_exposes_known_configs() -> None:
    names = {entry.name for entry in list_config_entries()}

    assert {
        "data",
        "csv_import",
        "data_quality",
        "features",
        "catalog",
        "factor_signals",
        "strategy",
        "model_strategy",
        "backtest",
        "model_backtest",
        "regime",
        "sensitivity",
        "attribution",
        "equity_comparison",
        "transformer",
        "prediction_signals",
        "studio_optimizer",
        "studio_profile",
        "studio_walk_forward",
        "studio_robustness",
        "studio_candidate_evidence",
    }.issubset(names)


def test_read_and_write_known_config(tmp_path: Path) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    (config_dir / "strategy.yaml").write_text(
        "score_column: return_20d\nstrategy_type: long_only_top_k\n",
        encoding="utf-8",
    )

    assert "score_column" in read_config_text("strategy", tmp_path)
    parsed = write_config_text(
        "strategy",
        "score_column: alpha101_001\nstrategy_type: long_only_top_k\n",
        tmp_path,
    )

    assert parsed["score_column"] == "alpha101_001"
    assert read_config("strategy", tmp_path)["score_column"] == "alpha101_001"


def test_config_editor_rejects_unknown_or_non_mapping_yaml(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown config"):
        read_config("not_allowed", tmp_path)

    with pytest.raises(ValueError, match="mapping"):
        write_config_text("strategy", "- just\n- a\n- list\n", tmp_path)
