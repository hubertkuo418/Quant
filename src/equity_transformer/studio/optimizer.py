from __future__ import annotations

import itertools
import json
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_transformer.reporting.equity_comparison import compare_equity_curves
from equity_transformer.studio.runner import StrategyRunResult, StrategyStudioRunner
from equity_transformer.studio.specs import (
    StrategySpec,
    load_strategy_spec,
    strategy_spec_from_dict,
)


@dataclass(frozen=True)
class Objective:
    metric: str
    direction: str = "maximize"


@dataclass(frozen=True)
class OptimizationConfig:
    base_spec_path: Path
    output_dir: Path
    search_space: dict[str, tuple[Any, ...]]
    objectives: tuple[Objective, ...]
    min_metrics: dict[str, float]
    max_metrics: dict[str, float]
    method: str = "grid"
    max_trials: int = 50
    random_seed: int = 42
    require_common_period: bool = True


def load_optimization_config(path: str | Path) -> OptimizationConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    config = OptimizationConfig(
        base_spec_path=Path(payload["base_spec_path"]),
        output_dir=Path(payload["output_dir"]),
        search_space={
            str(key): tuple(values)
            for key, values in payload.get("search_space", {}).items()
        },
        objectives=tuple(
            Objective(str(item["metric"]), str(item.get("direction", "maximize")))
            for item in payload.get(
                "objectives", [{"metric": "sharpe_ratio", "direction": "maximize"}]
            )
        ),
        min_metrics={
            str(key): float(value)
            for key, value in payload.get("min_metrics", {}).items()
        },
        max_metrics={
            str(key): float(value)
            for key, value in payload.get("max_metrics", {}).items()
        },
        method=str(payload.get("method", "grid")),
        max_trials=int(payload.get("max_trials", 50)),
        random_seed=int(payload.get("random_seed", 42)),
        require_common_period=bool(payload.get("require_common_period", True)),
    )
    _validate_optimization_config(config)
    return config


class StrategyOptimizer:
    def __init__(
        self,
        config: OptimizationConfig,
        runner: StrategyStudioRunner | None = None,
    ) -> None:
        _validate_optimization_config(config)
        self.config = config
        self.runner = runner or StrategyStudioRunner()

    def run(self, base_spec: StrategySpec | None = None) -> pd.DataFrame:
        base = base_spec or load_strategy_spec(self.config.base_spec_path)
        candidates = self._candidate_parameters()
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        run_results = []
        for trial, parameters in enumerate(candidates, start=1):
            spec = _apply_parameters(base, parameters, trial)
            result = self.runner.run(spec)
            run_results.append(result)
            rows.append(self._result_row(result, parameters, trial))
        results = pd.DataFrame(rows)
        results = self._align_common_period(results, run_results)
        results["feasible"] = results.apply(self._is_feasible, axis=1)
        results["pareto_efficient"] = _pareto_mask(
            results, self.config.objectives, results["feasible"]
        )
        primary = self.config.objectives[0]
        ascending = primary.direction == "minimize"
        results = results.sort_values(
            ["feasible", primary.metric],
            ascending=[False, ascending],
            na_position="last",
        ).reset_index(drop=True)
        self._save(results)
        return results

    def _align_common_period(
        self,
        results: pd.DataFrame,
        run_results: list[StrategyRunResult],
    ) -> pd.DataFrame:
        curve_paths = {
            result.run_id: result.run_dir / "backtest" / "equity_curve.csv"
            for result in run_results
        }
        if not all(path.exists() for path in curve_paths.values()):
            if self.config.require_common_period:
                raise FileNotFoundError(
                    "Optimizer requires every candidate equity curve for "
                    "common-period evaluation."
                )
            results["evaluation_basis"] = "individual_period"
            return results
        curves = {run_id: pd.read_csv(path) for run_id, path in curve_paths.items()}
        comparison, aligned = compare_equity_curves(curves)
        comparison = comparison.set_index("portfolio")
        identity = {"common_start", "common_end", "observations"}
        metric_columns = [
            column for column in comparison.columns if column not in identity
        ]
        for metric in metric_columns:
            if metric in results.columns:
                results[f"raw_{metric}"] = results[metric]
            results[metric] = results["run_id"].map(comparison[metric])
        results["common_start"] = results["run_id"].map(
            comparison["common_start"]
        )
        results["common_end"] = results["run_id"].map(comparison["common_end"])
        results["evaluation_basis"] = "common_period"
        comparison.reset_index().to_csv(
            self.config.output_dir / "common_period_comparison.csv", index=False
        )
        aligned_dir = self.config.output_dir / "aligned_curves"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        for run_id, frame in aligned.items():
            frame.to_csv(aligned_dir / f"{run_id}.csv", index=False)
        return results

    def _candidate_parameters(self) -> list[dict[str, Any]]:
        keys = list(self.config.search_space)
        combinations = [
            dict(zip(keys, values, strict=True))
            for values in itertools.product(
                *(self.config.search_space[key] for key in keys)
            )
        ]
        if self.config.method == "grid":
            return combinations[: self.config.max_trials]
        random.Random(self.config.random_seed).shuffle(combinations)
        return combinations[: self.config.max_trials]

    def _result_row(
        self,
        result: StrategyRunResult,
        parameters: dict[str, Any],
        trial: int,
    ) -> dict[str, Any]:
        return {
            "trial": trial,
            "run_id": result.run_id,
            "parameters": json.dumps(parameters, sort_keys=True),
            **result.metrics,
        }

    def _is_feasible(self, row: pd.Series) -> bool:
        for metric, minimum in self.config.min_metrics.items():
            if metric not in row or pd.isna(row[metric]) or row[metric] < minimum:
                return False
        for metric, maximum in self.config.max_metrics.items():
            if metric not in row or pd.isna(row[metric]) or row[metric] > maximum:
                return False
        return True

    def _save(self, results: pd.DataFrame) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        results.to_csv(self.config.output_dir / "results.csv", index=False)
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "base_spec_path": str(self.config.base_spec_path),
            "method": self.config.method,
            "trials": len(results),
            "feasible_trials": int(results["feasible"].sum()),
            "pareto_trials": int(results["pareto_efficient"].sum()),
            "objectives": [objective.__dict__ for objective in self.config.objectives],
            "min_metrics": self.config.min_metrics,
            "max_metrics": self.config.max_metrics,
            "require_common_period": self.config.require_common_period,
            "search_space": {
                key: list(values) for key, values in self.config.search_space.items()
            },
        }
        (self.config.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )


def _apply_parameters(
    base: StrategySpec,
    parameters: dict[str, Any],
    trial: int,
) -> StrategySpec:
    payload = base.to_dict()
    for dotted_key, value in parameters.items():
        target = payload
        parts = dotted_key.split(".")
        if not parts or any(not part for part in parts):
            raise ValueError(f"Invalid optimizer parameter path: {dotted_key}")
        for part in parts[:-1]:
            if part not in target or not isinstance(target[part], dict):
                raise ValueError(f"Unknown optimizer parameter path: {dotted_key}")
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown optimizer parameter path: {dotted_key}")
        target[parts[-1]] = value
    payload["version"] = f"{base.version}-trial{trial:03d}"
    candidate = strategy_spec_from_dict(payload)
    candidate.validate()
    return candidate


def _pareto_mask(
    results: pd.DataFrame,
    objectives: tuple[Objective, ...],
    feasible: pd.Series,
) -> pd.Series:
    mask = pd.Series(False, index=results.index)
    valid = results.loc[feasible].dropna(
        subset=[objective.metric for objective in objectives]
    )
    for index, candidate in valid.iterrows():
        dominated = False
        for other_index, other in valid.iterrows():
            if other_index == index:
                continue
            no_worse = []
            strictly_better = []
            for objective in objectives:
                left = other[objective.metric]
                right = candidate[objective.metric]
                if objective.direction == "maximize":
                    no_worse.append(left >= right)
                    strictly_better.append(left > right)
                else:
                    no_worse.append(left <= right)
                    strictly_better.append(left < right)
            if all(no_worse) and any(strictly_better):
                dominated = True
                break
        mask.loc[index] = not dominated
    return mask


def _validate_optimization_config(config: OptimizationConfig) -> None:
    if config.method not in {"grid", "random"}:
        raise ValueError("Optimizer method must be grid or random.")
    if not config.search_space or any(
        not values for values in config.search_space.values()
    ):
        raise ValueError("Optimizer search_space must contain non-empty values.")
    if config.max_trials <= 0:
        raise ValueError("Optimizer max_trials must be positive.")
    combinations = 1
    for values in config.search_space.values():
        combinations *= len(values)
    if config.require_common_period and min(combinations, config.max_trials) < 2:
        raise ValueError("Common-period optimization requires at least two trials.")
    if not config.objectives:
        raise ValueError("Optimizer requires at least one objective.")
    if any(
        objective.direction not in {"maximize", "minimize"}
        for objective in config.objectives
    ):
        raise ValueError("Objective direction must be maximize or minimize.")
