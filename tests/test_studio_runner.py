from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from equity_transformer.studio.registry import StrategyRunRegistry
from equity_transformer.studio.runner import StrategyStudioRunner
from equity_transformer.studio.specs import (
    ExecutionSpec,
    PortfolioSpec,
    SignalComponent,
    SignalSpec,
    StrategySpec,
    UniverseSpec,
)


def write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    dates = pd.bdate_range("2024-01-01", periods=12)
    rows = []
    for ticker, drift in (("AAA", 0.02), ("BBB", -0.01), ("SPY", 0.005)):
        price = 100.0
        for date in dates:
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
            price *= 1 + drift
    market_path = tmp_path / "market.parquet"
    pd.DataFrame(rows).to_parquet(market_path, index=False)

    signals = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "score": score}
            for date in dates
            for ticker, score in (("AAA", 1.0), ("BBB", -1.0), ("SPY", 2.0))
        ]
    )
    signal_path = tmp_path / "signals.parquet"
    signals.to_parquet(signal_path, index=False)
    return market_path, signal_path


def make_spec(tmp_path: Path) -> StrategySpec:
    market_path, signal_path = write_inputs(tmp_path)
    return StrategySpec(
        name="studio-test",
        version="1.0.0",
        description="runner integration",
        market_path=market_path,
        signal=SignalSpec(path=signal_path, score_column="score"),
        universe=UniverseSpec(excluded_tickers=("SPY",)),
        portfolio=PortfolioSpec(top_k=1, rebalance_frequency="W-FRI"),
        execution=ExecutionSpec(
            initial_capital=100_000,
            commission_bps=0,
            slippage_bps=0,
            execution_lag_days=0,
        ),
        benchmark_ticker="SPY",
    )


def test_studio_runner_persists_reproducible_run(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"

    result = StrategyStudioRunner(runs_root).run(make_spec(tmp_path))
    manifest = json.loads(
        (result.run_dir / "manifest.json").read_text(encoding="utf-8")
    )

    assert result.metrics["total_return"] > 0
    assert (result.run_dir / "strategy.yaml").exists()
    assert (result.run_dir / "target_weights.parquet").exists()
    assert (result.run_dir / "backtest" / "equity_curve.csv").exists()
    assert len(manifest["market_sha256"]) == 64
    assert len(manifest["signal_sha256"]) == 64
    assert manifest["spec_hash"] == make_spec(tmp_path).spec_hash


def test_run_registry_lists_and_reads_runs(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    result = StrategyStudioRunner(runs_root).run(make_spec(tmp_path))
    registry = StrategyRunRegistry(runs_root)

    records = registry.list_runs()
    loaded = registry.get(result.run_id)
    summary = registry.summary()

    assert len(records) == 1
    assert loaded.run_id == result.run_id
    assert summary.loc[0, "strategy"] == "studio-test"
    assert summary.loc[0, "sharpe_ratio"] == result.metrics["sharpe_ratio"]
    assert registry.find(strategy="studio-test", version="1.0.0")[0].run_id == (
        result.run_id
    )
    assert registry.find(strategy="missing") == []


def test_studio_runner_builds_weighted_component_score(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    signals = pd.read_parquet(spec.signal.path)
    signals["momentum"] = signals["score"]
    signals["risk"] = -signals["score"]
    signals.to_parquet(spec.signal.path, index=False)
    component_spec = StrategySpec(
        **{
            **spec.__dict__,
            "signal": SignalSpec(
                path=spec.signal.path,
                score_column="combined_score",
                components=(
                    SignalComponent("momentum", 1.0),
                    SignalComponent("risk", -0.5),
                ),
            ),
        }
    )

    result = StrategyStudioRunner(tmp_path / "component_runs").run(component_spec)
    weights = pd.read_parquet(result.run_dir / "target_weights.parquet")

    assert set(weights["ticker"]) == {"AAA"}
