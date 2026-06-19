from __future__ import annotations

from pathlib import Path

from equity_transformer.studio.optimizer import (
    Objective,
    OptimizationConfig,
    StrategyOptimizer,
)
from equity_transformer.studio.runner import StrategyRunResult
from equity_transformer.studio.specs import (
    PortfolioSpec,
    SignalSpec,
    StrategySpec,
)


class FakeRunner:
    def __init__(self, root: Path) -> None:
        self.root = root

    def run(self, spec: StrategySpec) -> StrategyRunResult:
        top_k = spec.portfolio.top_k
        metrics = (
            {
                "sharpe_ratio": 1.2,
                "annual_return": 0.3,
                "max_drawdown": -0.2,
                "average_turnover": 0.3,
            }
            if top_k == 5
            else {
                "sharpe_ratio": 0.9,
                "annual_return": 0.2,
                "max_drawdown": -0.1,
                "average_turnover": 0.1,
            }
        )
        return StrategyRunResult(
            run_id=f"top-{top_k}",
            run_dir=self.root / f"top-{top_k}",
            metrics=metrics,
        )


def make_spec(tmp_path: Path) -> StrategySpec:
    return StrategySpec(
        name="optimizer-test",
        version="1.0.0",
        description="",
        market_path=tmp_path / "market.parquet",
        signal=SignalSpec(tmp_path / "signals.parquet", "score"),
        portfolio=PortfolioSpec(top_k=5),
    )


def test_optimizer_applies_constraints_and_marks_pareto_candidates(
    tmp_path: Path,
) -> None:
    config = OptimizationConfig(
        base_spec_path=tmp_path / "base.yaml",
        output_dir=tmp_path / "optimization",
        search_space={"portfolio.top_k": (5, 10)},
        objectives=(
            Objective("sharpe_ratio", "maximize"),
            Objective("max_drawdown", "maximize"),
        ),
        min_metrics={"max_drawdown": -0.25},
        max_metrics={"average_turnover": 0.2},
        require_common_period=False,
    )

    results = StrategyOptimizer(config, FakeRunner(tmp_path)).run(make_spec(tmp_path))
    by_run = results.set_index("run_id")

    assert bool(by_run.loc["top-5", "feasible"]) is False
    assert bool(by_run.loc["top-10", "feasible"]) is True
    assert bool(by_run.loc["top-10", "pareto_efficient"]) is True
    assert (config.output_dir / "results.csv").exists()
    assert (config.output_dir / "manifest.json").exists()


def test_optimizer_rejects_unknown_parameter_path(tmp_path: Path) -> None:
    config = OptimizationConfig(
        base_spec_path=tmp_path / "base.yaml",
        output_dir=tmp_path / "optimization",
        search_space={"portfolio.not_real": (1,)},
        objectives=(Objective("sharpe_ratio"),),
        min_metrics={},
        max_metrics={},
        require_common_period=False,
    )

    try:
        StrategyOptimizer(config, FakeRunner(tmp_path)).run(make_spec(tmp_path))
    except ValueError as exc:
        assert "Unknown optimizer parameter path" in str(exc)
    else:
        raise AssertionError("Unknown optimizer paths should be rejected.")
