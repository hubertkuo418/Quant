from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_transformer.studio.specs import (
    StrategySpec,
    load_strategy_spec,
    save_strategy_spec,
)
from equity_transformer.studio.walk_forward import (
    WalkForwardConfig,
    WalkForwardEvaluator,
)

KNOWN_SCENARIOS = (
    "baseline",
    "double_costs",
    "lag_plus_one",
    "top_k_minus_two",
    "top_k_plus_two",
    "monthly_rebalance",
)


@dataclass(frozen=True)
class RobustnessConfig:
    strategy_spec_path: Path
    output_dir: Path
    scenarios: tuple[str, ...] = KNOWN_SCENARIOS
    train_days: int = 120
    test_days: int = 20
    step_days: int = 20
    purge_days: int = 5
    minimum_sharpe: float = 0.0
    maximum_drawdown: float = -0.20
    minimum_positive_fold_rate: float = 0.50

    def validate(self) -> None:
        unknown = set(self.scenarios).difference(KNOWN_SCENARIOS)
        if unknown:
            raise ValueError(f"Unknown robustness scenarios: {sorted(unknown)}")
        if not self.scenarios:
            raise ValueError("At least one robustness scenario is required.")
        if not 0 <= self.minimum_positive_fold_rate <= 1:
            raise ValueError("minimum_positive_fold_rate must be in [0, 1].")
        if not -1 <= self.maximum_drawdown <= 0:
            raise ValueError("maximum_drawdown must be in [-1, 0].")


@dataclass(frozen=True)
class RobustnessResult:
    output_dir: Path
    scenarios: pd.DataFrame
    aggregate: dict[str, float | bool | int]


def load_robustness_config(path: str | Path) -> RobustnessConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    config = RobustnessConfig(
        strategy_spec_path=Path(payload["strategy_spec_path"]),
        output_dir=Path(payload["output_dir"]),
        scenarios=tuple(payload.get("scenarios", KNOWN_SCENARIOS)),
        train_days=int(payload.get("train_days", 120)),
        test_days=int(payload.get("test_days", 20)),
        step_days=int(payload.get("step_days", payload.get("test_days", 20))),
        purge_days=int(payload.get("purge_days", 5)),
        minimum_sharpe=float(payload.get("minimum_sharpe", 0.0)),
        maximum_drawdown=float(payload.get("maximum_drawdown", -0.20)),
        minimum_positive_fold_rate=float(
            payload.get("minimum_positive_fold_rate", 0.50)
        ),
    )
    config.validate()
    return config


class RobustnessEvaluator:
    def __init__(self, config: RobustnessConfig) -> None:
        config.validate()
        self.config = config

    def run(self) -> RobustnessResult:
        base = load_strategy_spec(self.config.strategy_spec_path)
        output = self.config.output_dir
        output.mkdir(parents=True, exist_ok=True)
        rows = []
        for scenario_name in self.config.scenarios:
            scenario_spec = apply_robustness_scenario(base, scenario_name)
            spec_path = output / "scenario_specs" / f"{scenario_name}.yaml"
            save_strategy_spec(scenario_spec, spec_path)
            wf_result = WalkForwardEvaluator(
                WalkForwardConfig(
                    strategy_spec_path=spec_path,
                    output_dir=output / "walk_forward" / scenario_name,
                    train_days=self.config.train_days,
                    test_days=self.config.test_days,
                    step_days=self.config.step_days,
                    purge_days=self.config.purge_days,
                )
            ).run()
            fold_sharpe = pd.to_numeric(
                wf_result.folds["sharpe_ratio"], errors="coerce"
            )
            positive_fold_rate = float((fold_sharpe > 0).mean())
            metrics = wf_result.metrics
            passes = (
                metrics["sharpe_ratio"] >= self.config.minimum_sharpe
                and metrics["max_drawdown"] >= self.config.maximum_drawdown
                and positive_fold_rate >= self.config.minimum_positive_fold_rate
            )
            rows.append(
                {
                    "scenario": scenario_name,
                    "spec_hash": scenario_spec.spec_hash,
                    "folds": len(wf_result.folds),
                    "oos_observations": len(wf_result.equity_curve) - 1,
                    "total_return": metrics["total_return"],
                    "annual_return": metrics["annual_return"],
                    "sharpe_ratio": metrics["sharpe_ratio"],
                    "max_drawdown": metrics["max_drawdown"],
                    "average_turnover": metrics["average_turnover"],
                    "fold_sharpe_min": float(fold_sharpe.min()),
                    "fold_sharpe_std": float(fold_sharpe.std(ddof=0)),
                    "positive_fold_rate": positive_fold_rate,
                    "passes_constraints": passes,
                }
            )
        summary = pd.DataFrame(rows)
        aggregate = {
            "scenario_count": len(summary),
            "pass_rate": float(summary["passes_constraints"].mean()),
            "all_scenarios_pass": bool(summary["passes_constraints"].all()),
            "median_sharpe": float(summary["sharpe_ratio"].median()),
            "worst_sharpe": float(summary["sharpe_ratio"].min()),
            "sharpe_std": float(summary["sharpe_ratio"].std(ddof=0)),
            "worst_max_drawdown": float(summary["max_drawdown"].min()),
            "minimum_positive_fold_rate": float(
                summary["positive_fold_rate"].min()
            ),
        }
        summary.to_csv(output / "scenario_summary.csv", index=False)
        (output / "aggregate.json").write_text(
            json.dumps(aggregate, indent=2), encoding="utf-8"
        )
        (output / "report.md").write_text(
            _markdown_report(base, summary, aggregate), encoding="utf-8"
        )
        return RobustnessResult(output, summary, aggregate)


def apply_robustness_scenario(spec: StrategySpec, scenario: str) -> StrategySpec:
    if scenario not in KNOWN_SCENARIOS:
        raise ValueError(f"Unknown robustness scenario: {scenario}")
    version = f"{spec.version}-robust-{scenario.replace('_', '-')}"
    if scenario == "baseline":
        return replace(spec, version=version)
    if scenario == "double_costs":
        return replace(
            spec,
            version=version,
            execution=replace(
                spec.execution,
                commission_bps=spec.execution.commission_bps * 2,
                slippage_bps=spec.execution.slippage_bps * 2,
            ),
        )
    if scenario == "lag_plus_one":
        return replace(
            spec,
            version=version,
            execution=replace(
                spec.execution,
                execution_lag_days=spec.execution.execution_lag_days + 1,
            ),
        )
    if scenario == "top_k_minus_two":
        return replace(
            spec,
            version=version,
            portfolio=replace(spec.portfolio, top_k=max(1, spec.portfolio.top_k - 2)),
        )
    if scenario == "top_k_plus_two":
        return replace(
            spec,
            version=version,
            portfolio=replace(spec.portfolio, top_k=spec.portfolio.top_k + 2),
        )
    return replace(
        spec,
        version=version,
        portfolio=replace(spec.portfolio, rebalance_frequency="M"),
    )


def _markdown_report(
    spec: StrategySpec,
    summary: pd.DataFrame,
    aggregate: dict[str, float | bool | int],
) -> str:
    table_columns = [
        "scenario",
        "sharpe_ratio",
        "max_drawdown",
        "fold_sharpe_min",
        "positive_fold_rate",
        "passes_constraints",
    ]
    table = _markdown_table(summary, table_columns)
    return "\n".join(
        [
            f"# Robustness Report: {spec.name}",
            "",
            f"- Scenarios: {aggregate['scenario_count']}",
            f"- Pass rate: {aggregate['pass_rate']:.2%}",
            f"- Worst Sharpe: {aggregate['worst_sharpe']:.4f}",
            f"- Worst maximum drawdown: {aggregate['worst_max_drawdown']:.4f}",
            "",
            table,
            "",
            "Results use short frozen-strategy OOS folds and are not return "
            "guarantees.",
            "",
        ]
    )


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.select_dtypes(include="number"):
        selected[column] = selected[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])
