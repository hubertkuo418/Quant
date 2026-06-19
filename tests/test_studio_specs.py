from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from equity_transformer.studio.specs import (
    PortfolioSpec,
    RiskSpec,
    SignalComponent,
    SignalSpec,
    StrategySpec,
    load_strategy_spec,
    save_strategy_spec,
)


def make_spec(tmp_path: Path) -> StrategySpec:
    return StrategySpec(
        name="Momentum Top 3",
        version="1.0.0",
        description="test strategy",
        market_path=tmp_path / "market.parquet",
        signal=SignalSpec(
            path=tmp_path / "signals.parquet",
            score_column="score",
        ),
        portfolio=PortfolioSpec(top_k=3),
    )


def test_strategy_spec_yaml_round_trip_has_stable_hash(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    path = tmp_path / "strategy.yaml"

    save_strategy_spec(spec, path)
    loaded = load_strategy_spec(path)

    assert loaded == spec
    assert loaded.spec_hash == spec.spec_hash
    assert loaded.slug == "momentum-top-3"


def test_strategy_spec_hash_changes_with_portfolio_parameter(tmp_path: Path) -> None:
    spec = make_spec(tmp_path)
    changed = replace(spec, portfolio=replace(spec.portfolio, top_k=5))

    assert changed.spec_hash != spec.spec_hash


def test_strategy_spec_rejects_risk_weighting_without_volatility(
    tmp_path: Path,
) -> None:
    spec = replace(
        make_spec(tmp_path),
        portfolio=PortfolioSpec(weighting="risk_parity"),
    )

    with pytest.raises(ValueError, match="volatility_column"):
        spec.validate()


def test_strategy_spec_rejects_sector_cap_without_sector_data(tmp_path: Path) -> None:
    spec = replace(make_spec(tmp_path), risk=RiskSpec(max_sector_weight=0.3))

    with pytest.raises(ValueError, match="sector_column"):
        spec.validate()


def test_strategy_spec_round_trips_signal_components(tmp_path: Path) -> None:
    spec = replace(
        make_spec(tmp_path),
        signal=SignalSpec(
            path=tmp_path / "signals.parquet",
            score_column="custom_score",
            components=(
                SignalComponent("momentum_20d", 0.7),
                SignalComponent("volatility_20d", -0.3),
            ),
        ),
    )
    path = tmp_path / "components.yaml"

    save_strategy_spec(spec, path)
    loaded = load_strategy_spec(path)

    assert loaded.signal.components == spec.signal.components
