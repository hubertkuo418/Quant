from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from equity_transformer.gui.studio_wizard import (
    build_strategy_from_wizard,
    components_from_frame,
    components_to_frame,
    strategy_output_path,
    strategy_to_wizard_values,
)
from equity_transformer.studio.specs import SignalComponent, SignalSpec, StrategySpec


def make_spec(tmp_path: Path) -> StrategySpec:
    return StrategySpec(
        name="factor-top10",
        version="1.0.0",
        description="base",
        market_path=tmp_path / "market.parquet",
        signal=SignalSpec(
            path=tmp_path / "signals.parquet",
            score_column="factor_score",
            volatility_column="volatility_20d",
            components=(SignalComponent("return_20d", 1.0),),
        ),
    )


def test_wizard_round_trip_preserves_strategy(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    values = strategy_to_wizard_values(spec)
    rebuilt = build_strategy_from_wizard(spec, values, spec.signal.components)

    assert rebuilt == spec
    assert rebuilt.spec_hash == spec.spec_hash


def test_wizard_updates_portfolio_risk_and_universe(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    values = replace(
        strategy_to_wizard_values(spec),
        name="My Balanced Strategy",
        excluded_tickers="spy, AAPL\nMSFT, SPY",
        top_k=5,
        weighting="inverse_volatility",
        use_position_limit=True,
        max_position_weight=0.2,
        execution_lag_days=2,
    )

    rebuilt = build_strategy_from_wizard(spec, values, spec.signal.components)

    assert rebuilt.universe.excluded_tickers == ("SPY", "AAPL", "MSFT")
    assert rebuilt.portfolio.top_k == 5
    assert rebuilt.risk.max_position_weight == 0.2
    assert rebuilt.execution.execution_lag_days == 2
    assert strategy_output_path(rebuilt) == Path("strategies/my-balanced-strategy.yaml")


def test_component_editor_skips_blank_rows() -> None:
    original = (
        SignalComponent("return_20d", 0.7),
        SignalComponent("volatility_20d", -0.3, "cross_sectional_rank"),
    )
    frame = pd.concat(
        [components_to_frame(original), pd.DataFrame([{"column": ""}])],
        ignore_index=True,
    )

    assert components_from_frame(frame) == original


def test_component_editor_applies_defaults_to_new_rows() -> None:
    frame = pd.DataFrame([{"column": "return_5d", "weight": None}])

    assert components_from_frame(frame) == (
        SignalComponent("return_5d", 1.0, "cross_sectional_zscore"),
    )
