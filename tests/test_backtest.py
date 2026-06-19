from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.backtest.config import BacktestConfig
from equity_transformer.backtest.engine import BacktestEngine
from equity_transformer.backtest.metrics import (
    performance_metrics,
    relative_performance_metrics,
)


def make_market() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=5)
    rows = []
    for ticker, prices in {
        "AAA": [100, 110, 121, 133.1, 146.41],
        "BBB": [100, 100, 100, 100, 100],
    }.items():
        for date, price in zip(dates, prices, strict=True):
            rows.append({"date": date, "ticker": ticker, "adj_close": price})
    return pd.DataFrame(rows)


def make_market_with_volume() -> pd.DataFrame:
    market = make_market()
    market["volume"] = market["ticker"].map({"AAA": 1_000_000, "BBB": 1})
    return market


def make_weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")],
            "ticker": ["AAA"],
            "weight": [1.0],
        }
    )


def make_sector_weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "ticker": ["AAA", "BBB"],
            "sector": ["Technology", "Utilities"],
            "weight": [0.7, 0.3],
        }
    )


def make_config(tmp_path: Path, cost_bps: float = 0.0) -> BacktestConfig:
    return BacktestConfig(
        market_path=tmp_path / "market.parquet",
        weights_path=tmp_path / "weights.parquet",
        output_dir=tmp_path / "backtest",
        initial_capital=100.0,
        commission_bps=cost_bps,
        slippage_bps=0.0,
        annualization_factor=252,
        risk_free_rate=0.0,
    )


def test_backtest_nav_matches_known_path_without_costs(tmp_path: Path) -> None:
    engine = BacktestEngine(make_config(tmp_path))
    equity, holdings, metrics = engine.run(
        make_market(),
        make_weights(),
    )

    assert np.isclose(equity["nav"].iloc[-1], 146.41)
    assert not holdings.empty
    assert not engine.last_trade_log.empty
    assert not engine.last_exposure.empty
    assert metrics["total_return"] > 0
    assert metrics["average_gross_exposure"] > 0


def test_target_before_first_return_becomes_initial_position(tmp_path: Path) -> None:
    weights = make_weights().assign(date=pd.Timestamp("2024-01-01"))

    equity, holdings, _ = BacktestEngine(make_config(tmp_path)).run(
        make_market(), weights
    )

    assert np.isclose(equity["nav"].iloc[-1], 146.41)
    assert holdings["date"].min() == pd.Timestamp("2024-01-02")


def test_backtest_metrics_start_at_first_target_date(tmp_path: Path) -> None:
    weights = make_weights().assign(date=pd.Timestamp("2024-01-04"))

    equity, holdings, _ = BacktestEngine(make_config(tmp_path)).run(
        make_market(), weights
    )

    assert equity["date"].iloc[0] == pd.Timestamp("2024-01-03")
    assert equity["date"].iloc[1] == pd.Timestamp("2024-01-04")
    assert holdings["date"].min() == pd.Timestamp("2024-01-04")


def test_backtest_rejects_empty_target_weights(tmp_path: Path) -> None:
    empty = pd.DataFrame(columns=["date", "ticker", "weight"])

    with np.testing.assert_raises_regex(ValueError, "No target weights remain"):
        BacktestEngine(make_config(tmp_path)).run(make_market(), empty)


def test_transaction_cost_reduces_nav(tmp_path: Path) -> None:
    no_cost, _, _ = BacktestEngine(make_config(tmp_path / "a")).run(
        make_market(),
        make_weights(),
    )
    with_cost, _, _ = BacktestEngine(make_config(tmp_path / "b", 100)).run(
        make_market(),
        make_weights(),
    )

    assert with_cost["nav"].iloc[-1] < no_cost["nav"].iloc[-1]


def test_cash_interest_increases_partially_invested_portfolio_nav(
    tmp_path: Path,
) -> None:
    weights = make_weights().assign(weight=0.5)
    base_config = make_config(tmp_path / "base")
    cash_config = BacktestConfig(
        **{
            **make_config(tmp_path / "cash").__dict__,
            "annual_cash_rate": 0.252,
        }
    )

    base, _, _ = BacktestEngine(base_config).run(make_market(), weights)
    with_cash, _, metrics = BacktestEngine(cash_config).run(make_market(), weights)

    assert with_cash["nav"].iloc[-1] > base["nav"].iloc[-1]
    assert metrics["total_cash_interest"] > 0


def test_short_borrow_cost_reduces_long_short_portfolio_nav(tmp_path: Path) -> None:
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02")] * 2,
            "ticker": ["AAA", "BBB"],
            "weight": [-0.5, 0.5],
        }
    )
    base_config = make_config(tmp_path / "base")
    borrow_config = BacktestConfig(
        **{
            **make_config(tmp_path / "borrow").__dict__,
            "annual_borrow_rate": 0.252,
        }
    )

    base, _, _ = BacktestEngine(base_config).run(make_market(), weights)
    with_borrow, _, metrics = BacktestEngine(borrow_config).run(
        make_market(), weights
    )

    assert with_borrow["nav"].iloc[-1] < base["nav"].iloc[-1]
    assert metrics["total_borrow_cost"] > 0


def test_execution_lag_delays_target_weight_application(tmp_path: Path) -> None:
    config = BacktestConfig(
        **{
            **make_config(tmp_path).__dict__,
            "execution_lag_days": 1,
        }
    )
    _, no_lag_holdings, _ = BacktestEngine(make_config(tmp_path / "no_lag")).run(
        make_market(), make_weights()
    )
    equity, holdings, _ = BacktestEngine(config).run(make_market(), make_weights())

    assert no_lag_holdings["date"].min() == pd.Timestamp("2024-01-02")
    assert equity.loc[0, "nav"] == 100.0
    assert holdings["date"].min() == pd.Timestamp("2024-01-03")


def test_liquidity_filter_removes_illiquid_positions(tmp_path: Path) -> None:
    weights = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-02")],
            "ticker": ["AAA", "BBB"],
            "weight": [0.5, 0.5],
        }
    )
    config = BacktestConfig(
        **{
            **make_config(tmp_path).__dict__,
            "min_dollar_volume": 10_000,
            "liquidity_window": 1,
        }
    )
    _, holdings, _ = BacktestEngine(config).run(make_market_with_volume(), weights)

    assert set(holdings["ticker"]) == {"AAA"}


def test_backtest_writes_artifacts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    BacktestEngine(config).run(make_market(), make_weights())

    assert (config.output_dir / "equity_curve.csv").exists()
    assert (config.output_dir / "holdings.parquet").exists()
    assert (config.output_dir / "trade_log.parquet").exists()
    assert (config.output_dir / "exposure.csv").exists()
    assert (config.output_dir / "sector_exposure.csv").exists()
    assert (config.output_dir / "metrics.json").exists()


def test_backtest_records_sector_exposure_when_weights_include_sector(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    engine = BacktestEngine(config)
    _, holdings, _ = engine.run(make_market(), make_sector_weights())

    assert "sector" in holdings.columns
    assert set(holdings["sector"]) == {"Technology", "Utilities"}
    assert not engine.last_sector_exposure.empty
    assert set(engine.last_sector_exposure["sector"]) == {"Technology", "Utilities"}
    assert (config.output_dir / "sector_exposure.csv").exists()


def test_performance_metrics_handle_drawdown() -> None:
    equity = pd.DataFrame(
        {
            "nav": [100.0, 110.0, 99.0, 120.0],
            "net_return": [0.0, 0.1, -0.1, 120 / 99 - 1],
            "turnover": [0.0, 1.0, 0.0, 0.0],
            "cost": [0.0, 0.0, 0.0, 0.0],
        }
    )

    exposure = pd.DataFrame(
        {
            "gross_exposure": [0.0, 1.0, 1.0, 1.0],
            "net_exposure": [0.0, 1.0, 1.0, 1.0],
            "active_positions": [0, 1, 1, 1],
        }
    )
    metrics = performance_metrics(
        equity,
        annualization_factor=252,
        risk_free_rate=0,
        exposure=exposure,
    )

    assert np.isclose(metrics["total_return"], 0.2)
    assert metrics["max_drawdown"] < 0
    assert metrics["average_turnover"] == 0.25
    assert metrics["average_gross_exposure"] == 0.75
    assert np.isclose(metrics["value_at_risk_95"], 0.085)
    assert np.isclose(metrics["conditional_value_at_risk_95"], 0.1)
    assert metrics["profit_factor"] > 0


def test_relative_performance_metrics_compute_active_risk() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    strategy = pd.Series([0.01, 0.02, -0.01, 0.03], index=dates)
    benchmark = pd.Series([0.005, 0.01, -0.005, 0.01], index=dates)

    metrics = relative_performance_metrics(strategy, benchmark, 252, 0.0)

    assert metrics["tracking_error"] > 0
    assert metrics["information_ratio"] > 0
    assert np.isfinite(metrics["beta"])
    assert np.isfinite(metrics["annual_alpha"])


def test_backtest_reports_metrics_relative_to_configured_benchmark(
    tmp_path: Path,
) -> None:
    config = BacktestConfig(
        **{
            **make_config(tmp_path).__dict__,
            "benchmark_ticker": "BBB",
        }
    )

    _, _, metrics = BacktestEngine(config).run(make_market(), make_weights())

    assert {"tracking_error", "information_ratio", "beta", "annual_alpha"}.issubset(
        metrics
    )


def test_backtest_rejects_missing_benchmark_ticker(tmp_path: Path) -> None:
    config = BacktestConfig(
        **{
            **make_config(tmp_path).__dict__,
            "benchmark_ticker": "MISSING",
        }
    )

    with np.testing.assert_raises_regex(ValueError, "Benchmark ticker not found"):
        BacktestEngine(config).run(make_market(), make_weights())
