from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from equity_transformer.studio.comparison import compare_strategy_runs
from equity_transformer.studio.registry import StrategyRunRegistry


def write_run(root: Path, run_id: str, start: str, returns: list[float]) -> None:
    run_dir = root / run_id
    backtest = run_dir / "backtest"
    backtest.mkdir(parents=True)
    dates = pd.bdate_range(start, periods=len(returns))
    frame = pd.DataFrame({"date": dates, "net_return": returns})
    frame["nav"] = (1 + frame["net_return"]).cumprod()
    frame.to_csv(backtest / "equity_curve.csv", index=False)
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_id, "strategy_name": run_id, "metrics": {}}),
        encoding="utf-8",
    )


def test_compare_strategy_runs_uses_common_period(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    write_run(runs, "run-a", "2024-01-01", [0.0, 0.01, 0.01, 0.01, 0.01])
    write_run(runs, "run-b", "2024-01-03", [0.0, -0.01, 0.02, 0.01])

    comparison = compare_strategy_runs(
        ["run-a", "run-b"],
        StrategyRunRegistry(runs),
        tmp_path / "comparison",
    )

    assert set(comparison["portfolio"]) == {"run-a", "run-b"}
    assert comparison["common_start"].nunique() == 1
    assert (tmp_path / "comparison" / "comparison.csv").exists()
    assert (tmp_path / "comparison" / "aligned_curves" / "run-a.csv").exists()


def test_compare_strategy_runs_requires_two_unique_runs(tmp_path: Path) -> None:
    registry = StrategyRunRegistry(tmp_path / "runs")

    try:
        compare_strategy_runs(["same", "same"], registry, tmp_path / "output")
    except ValueError as exc:
        assert "two unique" in str(exc)
    else:
        raise AssertionError("Duplicate run IDs should be rejected.")
