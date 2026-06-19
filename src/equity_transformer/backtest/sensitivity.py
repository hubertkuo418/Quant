from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_transformer.backtest.config import BacktestConfig, load_backtest_config
from equity_transformer.backtest.engine import BacktestEngine

SENSITIVITY_FIELDS = {
    "commission_bps",
    "slippage_bps",
    "execution_lag_days",
    "min_dollar_volume",
    "liquidity_window",
    "annual_cash_rate",
    "annual_borrow_rate",
}


@dataclass(frozen=True)
class SensitivityScenario:
    name: str
    overrides: dict[str, float | int | None]


@dataclass(frozen=True)
class SensitivityConfig:
    base_backtest_config: Path
    output_dir: Path
    scenarios: tuple[SensitivityScenario, ...]


def load_sensitivity_config(path: str | Path) -> SensitivityConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    scenarios = tuple(
        SensitivityScenario(
            name=str(item["name"]),
            overrides=dict(item.get("overrides", {})),
        )
        for item in payload["scenarios"]
    )
    return SensitivityConfig(
        base_backtest_config=Path(payload["base_backtest_config"]),
        output_dir=Path(payload["output_dir"]),
        scenarios=scenarios,
    )


class SensitivityPipeline:
    def __init__(self, config: SensitivityConfig) -> None:
        self.config = config

    def run(
        self,
        market: pd.DataFrame | None = None,
        target_weights: pd.DataFrame | None = None,
        base_config: BacktestConfig | None = None,
    ) -> pd.DataFrame:
        base = base_config or load_backtest_config(self.config.base_backtest_config)
        market_frame = (
            market.copy() if market is not None else pd.read_parquet(base.market_path)
        )
        weights = (
            target_weights.copy()
            if target_weights is not None
            else pd.read_parquet(base.weights_path)
        )
        self._validate_scenarios()
        rows = []
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for scenario in self.config.scenarios:
            scenario_config = replace(
                base,
                output_dir=self.config.output_dir / scenario.name,
                **scenario.overrides,
            )
            equity, _, metrics = BacktestEngine(scenario_config).run(
                market_frame,
                weights,
            )
            rows.append(
                {
                    "scenario": scenario.name,
                    **scenario.overrides,
                    "ending_nav": float(equity["nav"].iloc[-1]),
                    **metrics,
                }
            )

        comparison = pd.DataFrame(rows)
        comparison.to_csv(self.config.output_dir / "comparison.csv", index=False)
        (self.config.output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "base_backtest_config": str(self.config.base_backtest_config),
                    "scenarios": [
                        {"name": item.name, "overrides": item.overrides}
                        for item in self.config.scenarios
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return comparison

    def _validate_scenarios(self) -> None:
        names = [scenario.name for scenario in self.config.scenarios]
        if not names:
            raise ValueError("At least one sensitivity scenario is required.")
        if len(names) != len(set(names)):
            raise ValueError("Sensitivity scenario names must be unique.")
        for scenario in self.config.scenarios:
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", scenario.name) is None:
                raise ValueError(f"Invalid sensitivity scenario name: {scenario.name}")
            unknown = set(scenario.overrides).difference(SENSITIVITY_FIELDS)
            if unknown:
                raise ValueError(
                    f"Scenario {scenario.name} has unsupported overrides: "
                    f"{sorted(unknown)}"
                )
            self._validate_override_values(scenario)

    @staticmethod
    def _validate_override_values(scenario: SensitivityScenario) -> None:
        non_negative = {
            "commission_bps",
            "slippage_bps",
            "execution_lag_days",
            "min_dollar_volume",
            "annual_cash_rate",
            "annual_borrow_rate",
        }
        for key in non_negative.intersection(scenario.overrides):
            value = scenario.overrides[key]
            if value is not None and value < 0:
                raise ValueError(f"Scenario {scenario.name} has negative {key}.")
        if "liquidity_window" in scenario.overrides:
            value = scenario.overrides["liquidity_window"]
            if value is None or value < 1:
                raise ValueError(
                    f"Scenario {scenario.name} requires positive liquidity_window."
                )
