from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.backtest.benchmark import (
    BenchmarkConfig,
    BenchmarkPipeline,
    buy_and_hold_weights,
    equal_weight_universe_weights,
    momentum_top_k_weights,
)


def make_market(days: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    for ticker_index, ticker in enumerate(("AAA", "BBB", "SPY")):
        for day_index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": 100 + ticker_index * 5 + day_index,
                }
            )
    return pd.DataFrame(rows)


def make_config(tmp_path: Path) -> BenchmarkConfig:
    return BenchmarkConfig(
        market_path=tmp_path / "market.parquet",
        output_dir=tmp_path / "benchmarks",
        rebalance_frequency="W-FRI",
        buy_and_hold_ticker="SPY",
        momentum_window=3,
        momentum_top_k=1,
        initial_capital=100.0,
        commission_bps=0.0,
        slippage_bps=0.0,
        annualization_factor=252,
        risk_free_rate=0.0,
    )


def test_equal_weight_benchmark_sums_to_one_by_date() -> None:
    weights = equal_weight_universe_weights(make_market(), "W-FRI")

    assert weights["date"].nunique() == 2
    sums = weights.groupby("date")["weight"].sum()
    assert np.allclose(sums.to_numpy(), 1.0)


def test_buy_and_hold_weight_starts_once() -> None:
    weights = buy_and_hold_weights(make_market(), "SPY")

    assert len(weights) == 1
    assert weights.loc[0, "ticker"] == "SPY"
    assert weights.loc[0, "weight"] == 1.0


def test_momentum_top_k_weights_selects_winners() -> None:
    weights = momentum_top_k_weights(make_market(days=8), 3, 1, "W-FRI")

    assert not weights.empty
    assert weights.groupby("date")["weight"].sum().eq(1.0).all()
    assert weights.groupby("date")["ticker"].nunique().eq(1).all()


def test_benchmark_pipeline_writes_comparison(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    comparison = BenchmarkPipeline(config).run(make_market())

    assert set(comparison["benchmark"]) == {
        "equal_weight_universe",
        "buy_and_hold_SPY",
        "momentum_3d_top_1",
    }
    assert (config.output_dir / "comparison.csv").exists()
    assert (config.output_dir / "equal_weight_universe").exists()
    buy_and_hold = comparison.set_index("benchmark").loc["buy_and_hold_SPY"]
    assert buy_and_hold["total_return"] > 0
