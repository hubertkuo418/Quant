from __future__ import annotations

from pathlib import Path

import pandas as pd

from equity_transformer.studio.robustness import (
    RobustnessConfig,
    RobustnessEvaluator,
    apply_robustness_scenario,
)
from equity_transformer.studio.specs import (
    ExecutionSpec,
    PortfolioSpec,
    SignalSpec,
    StrategySpec,
    save_strategy_spec,
)


def make_spec(tmp_path: Path) -> Path:
    dates = pd.bdate_range("2024-01-01", periods=55)
    market = []
    signals = []
    for ticker, drift, score in (
        ("AAA", 0.01, 1.0),
        ("BBB", -0.005, -1.0),
        ("SPY", 0.002, 0.0),
    ):
        price = 100.0
        for date in dates:
            market.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
            signals.append({"date": date, "ticker": ticker, "score": score})
            price *= 1 + drift
    market_path = tmp_path / "market.parquet"
    signal_path = tmp_path / "signals.parquet"
    pd.DataFrame(market).to_parquet(market_path, index=False)
    pd.DataFrame(signals).to_parquet(signal_path, index=False)
    spec = StrategySpec(
        name="robustness-test",
        version="1.0.0",
        description="test",
        market_path=market_path,
        signal=SignalSpec(signal_path, "score"),
        portfolio=PortfolioSpec(top_k=3, rebalance_frequency="W-FRI"),
        execution=ExecutionSpec(100_000, 1, 1, 1),
        benchmark_ticker="SPY",
    )
    path = tmp_path / "strategy.yaml"
    save_strategy_spec(spec, path)
    return path


def test_apply_robustness_scenarios_changes_expected_parameters(tmp_path: Path) -> None:
    spec_path = make_spec(tmp_path)
    from equity_transformer.studio.specs import load_strategy_spec

    spec = load_strategy_spec(spec_path)

    assert apply_robustness_scenario(spec, "double_costs").execution.commission_bps == 2
    lagged = apply_robustness_scenario(spec, "lag_plus_one")
    assert lagged.execution.execution_lag_days == 2
    assert apply_robustness_scenario(spec, "top_k_minus_two").portfolio.top_k == 1
    monthly = apply_robustness_scenario(spec, "monthly_rebalance")
    assert monthly.portfolio.rebalance_frequency == "M"


def test_robustness_evaluator_writes_scenario_and_aggregate_results(
    tmp_path: Path,
) -> None:
    config = RobustnessConfig(
        strategy_spec_path=make_spec(tmp_path),
        output_dir=tmp_path / "robustness",
        scenarios=("baseline", "double_costs", "lag_plus_one"),
        train_days=20,
        test_days=10,
        step_days=10,
        purge_days=2,
    )

    result = RobustnessEvaluator(config).run()

    assert len(result.scenarios) == 3
    assert set(result.scenarios["scenario"]) == set(config.scenarios)
    assert result.scenarios["oos_observations"].nunique() == 1
    assert 0 <= result.aggregate["pass_rate"] <= 1
    assert (result.output_dir / "scenario_summary.csv").exists()
    assert (result.output_dir / "report.md").exists()
