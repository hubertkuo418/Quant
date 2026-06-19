from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.studio.specs import (
    ExecutionSpec,
    PortfolioSpec,
    SignalSpec,
    StrategySpec,
    save_strategy_spec,
)
from equity_transformer.studio.walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
    build_walk_forward_folds,
)


def write_inputs(tmp_path: Path, periods: int = 70) -> tuple[Path, Path]:
    dates = pd.bdate_range("2024-01-01", periods=periods)
    market_rows = []
    signal_rows = []
    for ticker, drift, score in (
        ("AAA", 0.01, 1.0),
        ("BBB", -0.005, -1.0),
        ("SPY", 0.002, 0.0),
    ):
        price = 100.0
        for date in dates:
            market_rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
            signal_rows.append({"date": date, "ticker": ticker, "score": score})
            price *= 1 + drift
    market_path = tmp_path / "market.parquet"
    signal_path = tmp_path / "signals.parquet"
    pd.DataFrame(market_rows).to_parquet(market_path, index=False)
    pd.DataFrame(signal_rows).to_parquet(signal_path, index=False)
    return market_path, signal_path


def make_spec(tmp_path: Path) -> Path:
    market_path, signal_path = write_inputs(tmp_path)
    spec = StrategySpec(
        name="walk-forward-test",
        version="1.0.0",
        description="test",
        market_path=market_path,
        signal=SignalSpec(signal_path, "score"),
        portfolio=PortfolioSpec(top_k=1, rebalance_frequency="W-FRI"),
        execution=ExecutionSpec(
            initial_capital=100_000,
            commission_bps=0,
            slippage_bps=0,
            execution_lag_days=1,
        ),
        benchmark_ticker="SPY",
    )
    path = tmp_path / "strategy.yaml"
    save_strategy_spec(spec, path)
    return path


def test_build_walk_forward_folds_respects_purge_and_rolling_window() -> None:
    dates = pd.bdate_range("2024-01-01", periods=50)
    config = WalkForwardConfig(Path("spec"), Path("output"), 20, 10, 10, 2)

    folds = build_walk_forward_folds(dates, config)

    assert len(folds) == 2
    assert folds[0].train_start == dates[0]
    assert folds[0].train_end == dates[19]
    assert folds[0].test_start == dates[22]
    assert folds[1].train_start == dates[10]


def test_walk_forward_rejects_overlapping_test_windows() -> None:
    config = WalkForwardConfig(Path("spec"), Path("output"), 20, 10, 5, 1)

    with pytest.raises(ValueError, match="avoid overlap"):
        config.validate()


def test_walk_forward_evaluator_persists_stitched_oos_results(tmp_path: Path) -> None:
    config = WalkForwardConfig(
        strategy_spec_path=make_spec(tmp_path),
        output_dir=tmp_path / "walk_forward",
        train_days=20,
        test_days=10,
        step_days=10,
        purge_days=2,
    )

    result = WalkForwardEvaluator(config).run()
    manifest = json.loads(
        (result.output_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert len(result.folds) == 4
    assert result.equity_curve["date"].is_monotonic_increasing
    assert result.equity_curve["date"].is_unique
    assert result.metrics["total_return"] > 0
    assert manifest["evaluation_mode"] == "frozen_strategy_rolling_oos"
    assert (result.output_dir / "oos_equity_curve.csv").exists()


def test_walk_forward_honors_explicit_first_test_date(tmp_path: Path) -> None:
    config = WalkForwardConfig(
        strategy_spec_path=make_spec(tmp_path),
        output_dir=tmp_path / "bounded_walk_forward",
        train_days=20,
        test_days=10,
        step_days=10,
        purge_days=2,
        first_test_date="2024-02-20",
    )

    result = WalkForwardEvaluator(config).run()

    assert pd.to_datetime(result.folds["test_start"]).min() >= pd.Timestamp(
        "2024-02-20"
    )
