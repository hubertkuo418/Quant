from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.backtest.regime import (
    RegimeAnalysisConfig,
    RegimeAnalysisPipeline,
    classify_market_regimes,
    summarize_regime_performance,
)


def make_market(days: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=days)
    returns = np.where(
        np.arange(days) < days // 2,
        0.001 + np.sin(np.arange(days)) * 0.002,
        -0.0005 + np.sin(np.arange(days)) * 0.012,
    )
    prices = 100 * np.cumprod(1 + returns)
    return pd.DataFrame({"date": dates, "ticker": "SPY", "adj_close": prices})


def test_regime_classifier_uses_lagged_volatility_threshold() -> None:
    market = make_market()
    regimes = classify_market_regimes(
        market,
        benchmark_ticker="SPY",
        trend_window=10,
        volatility_window=5,
        threshold_window=20,
    )

    row = regimes.iloc[10]
    source = market.sort_values("date").copy()
    source["return"] = source["adj_close"].pct_change()
    source["volatility"] = source["return"].rolling(5).std() * np.sqrt(252)
    expected = source.loc[
        source["date"] < row["date"], "volatility"
    ].tail(20).median()

    assert not regimes.empty
    assert set(regimes["trend_regime"]).issubset({"bull", "bear"})
    assert set(regimes["volatility_regime"]) == {"high_vol", "low_vol"}
    assert np.isclose(row["volatility_threshold"], expected)


def test_regime_classifier_rejects_unknown_benchmark() -> None:
    with np.testing.assert_raises_regex(ValueError, "Benchmark ticker not found"):
        classify_market_regimes(make_market(), "QQQ", 10, 5, 20)


def test_regime_performance_summarizes_each_regime() -> None:
    dates = pd.bdate_range("2024-01-01", periods=6)
    equity = pd.DataFrame(
        {
            "date": dates,
            "net_return": [0.01, 0.02, -0.01, -0.02, 0.01, 0.03],
        }
    )
    regimes = pd.DataFrame(
        {
            "date": dates,
            "regime": ["bull_low_vol"] * 3 + ["bear_high_vol"] * 3,
        }
    )

    summary = summarize_regime_performance(equity, regimes)

    assert set(summary["regime"]) == {"bull_low_vol", "bear_high_vol"}
    assert summary["observations"].sum() == 6
    assert {"sharpe_ratio", "max_drawdown", "win_rate"}.issubset(summary.columns)


def test_regime_pipeline_writes_artifacts(tmp_path: Path) -> None:
    market = make_market()
    dates = pd.to_datetime(market["date"].drop_duplicates())
    equity = pd.DataFrame({"date": dates, "net_return": 0.001})
    config = RegimeAnalysisConfig(
        market_path=tmp_path / "market.parquet",
        equity_path=tmp_path / "equity.csv",
        output_dir=tmp_path / "regimes",
        benchmark_ticker="SPY",
        trend_window=10,
        volatility_window=5,
        threshold_window=20,
        annualization_factor=252,
    )

    regimes, performance = RegimeAnalysisPipeline(config).run(market, equity)

    assert not regimes.empty
    assert not performance.empty
    assert (config.output_dir / "daily_regimes.csv").exists()
    assert (config.output_dir / "regime_performance.csv").exists()
    assert (config.output_dir / "manifest.json").exists()
