from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.alphas.config import AlphaConfig
from equity_transformer.alphas.pipeline import AlphaPipeline
from equity_transformer.alphas.registry import alpha_registry


def make_market(rows: int = 30) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    parts = []
    for ticker, offset in [("AAA", 0.0), ("BBB", 20.0)]:
        close = 100 + offset + np.arange(rows, dtype=float)
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000 + np.arange(rows) * 5,
                }
            )
        )
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"])


def make_config(tmp_path: Path) -> AlphaConfig:
    return AlphaConfig(
        market_path=tmp_path / "market.parquet",
        output_path=tmp_path / "alphas.parquet",
        metadata_path=tmp_path / "alpha_manifest.json",
        alphas=(
            "alpha_momentum_20d",
            "alpha_reversal_5d",
            "alpha_price_position_20d",
        ),
    )


def test_alpha_registry_contains_starter_alphas() -> None:
    registry = alpha_registry()

    assert "alpha_momentum_20d" in registry
    assert registry["alpha_momentum_20d"].family == "momentum"
    assert "alpha101_001" in registry
    assert registry["alpha101_001"].family == "alpha101"


def test_alpha_pipeline_outputs_selected_columns(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    panel = AlphaPipeline(config).run(make_market())
    aaa = panel[panel["ticker"] == "AAA"].reset_index(drop=True)

    assert set(config.alphas).issubset(panel.columns)
    assert np.isclose(aaa.loc[20, "alpha_momentum_20d"], 120 / 100 - 1)
    assert np.isclose(aaa.loc[5, "alpha_reversal_5d"], -(105 / 100 - 1))
    assert config.output_path.exists()
    assert config.metadata_path.exists()


def test_alpha101_batch_outputs_expected_columns(tmp_path: Path) -> None:
    config = AlphaConfig(
        **{
            **make_config(tmp_path).__dict__,
            "alphas": tuple(f"alpha101_{index:03d}" for index in range(1, 21)),
        }
    )
    panel = AlphaPipeline(config).run(make_market(90))

    assert set(config.alphas).issubset(panel.columns)
    assert panel["alpha101_001"].notna().any()
    assert panel["alpha101_010"].notna().any()
    assert panel["alpha101_011"].notna().any()
    assert panel["alpha101_020"].notna().any()
    finite = panel[list(config.alphas)].replace([np.inf, -np.inf], np.nan)
    assert finite.notna().sum().gt(0).all()


def test_alpha_pipeline_rejects_unknown_alpha(tmp_path: Path) -> None:
    config = AlphaConfig(
        **{
            **make_config(tmp_path).__dict__,
            "alphas": ("does_not_exist",),
        }
    )

    try:
        AlphaPipeline(config).run(make_market())
    except ValueError as exc:
        assert "Unknown alphas" in str(exc)
    else:
        raise AssertionError("Expected unknown alpha to raise ValueError.")
