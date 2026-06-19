from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from equity_transformer.studio.specs import (
    ExecutionSpec,
    PortfolioSpec,
    RiskSpec,
    SignalComponent,
    SignalSpec,
    StrategySpec,
    UniverseSpec,
)


@dataclass(frozen=True)
class StrategyWizardValues:
    name: str
    version: str
    description: str
    market_path: str
    signal_path: str
    score_column: str
    volatility_column: str
    sector_column: str
    excluded_tickers: str
    start_date: str
    end_date: str
    strategy_type: str
    top_k: int
    long_quantile: float
    short_quantile: float
    weighting: str
    rebalance_frequency: str
    use_position_limit: bool
    max_position_weight: float
    use_sector_limit: bool
    max_sector_weight: float
    use_liquidity_limit: bool
    min_dollar_volume: float
    liquidity_window: int
    initial_capital: float
    commission_bps: float
    slippage_bps: float
    execution_lag_days: int
    annual_cash_rate: float
    annual_borrow_rate: float
    benchmark_ticker: str
    risk_free_rate: float


def strategy_to_wizard_values(spec: StrategySpec) -> StrategyWizardValues:
    return StrategyWizardValues(
        name=spec.name,
        version=spec.version,
        description=spec.description,
        market_path=str(spec.market_path),
        signal_path=str(spec.signal.path),
        score_column=spec.signal.score_column,
        volatility_column=spec.signal.volatility_column or "",
        sector_column=spec.signal.sector_column or "",
        excluded_tickers=", ".join(spec.universe.excluded_tickers),
        start_date=spec.universe.start_date or "",
        end_date=spec.universe.end_date or "",
        strategy_type=spec.portfolio.strategy_type,
        top_k=spec.portfolio.top_k,
        long_quantile=spec.portfolio.long_quantile,
        short_quantile=spec.portfolio.short_quantile,
        weighting=spec.portfolio.weighting,
        rebalance_frequency=spec.portfolio.rebalance_frequency,
        use_position_limit=spec.risk.max_position_weight is not None,
        max_position_weight=spec.risk.max_position_weight or 0.1,
        use_sector_limit=spec.risk.max_sector_weight is not None,
        max_sector_weight=spec.risk.max_sector_weight or 0.3,
        use_liquidity_limit=spec.risk.min_dollar_volume is not None,
        min_dollar_volume=spec.risk.min_dollar_volume or 1_000_000.0,
        liquidity_window=spec.risk.liquidity_window,
        initial_capital=spec.execution.initial_capital,
        commission_bps=spec.execution.commission_bps,
        slippage_bps=spec.execution.slippage_bps,
        execution_lag_days=spec.execution.execution_lag_days,
        annual_cash_rate=spec.execution.annual_cash_rate,
        annual_borrow_rate=spec.execution.annual_borrow_rate,
        benchmark_ticker=spec.benchmark_ticker or "",
        risk_free_rate=spec.risk_free_rate,
    )


def components_to_frame(components: tuple[SignalComponent, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "column": component.column,
                "weight": component.weight,
                "transform": component.transform,
            }
            for component in components
        ],
        columns=["column", "weight", "transform"],
    )


def components_from_frame(frame: pd.DataFrame) -> tuple[SignalComponent, ...]:
    components = []
    for record in frame.to_dict(orient="records"):
        column = str(record.get("column", "")).strip()
        if not column or column.lower() == "nan":
            continue
        raw_weight = record.get("weight")
        weight = 1.0 if pd.isna(raw_weight) else float(raw_weight)
        raw_transform = record.get("transform")
        transform = (
            "cross_sectional_zscore"
            if pd.isna(raw_transform) or not str(raw_transform).strip()
            else str(raw_transform).strip()
        )
        components.append(SignalComponent(column, weight, transform))
    return tuple(components)


def build_strategy_from_wizard(
    base: StrategySpec,
    values: StrategyWizardValues,
    components: tuple[SignalComponent, ...] = (),
) -> StrategySpec:
    spec = replace(
        base,
        name=values.name.strip(),
        version=values.version.strip(),
        description=values.description.strip(),
        market_path=Path(values.market_path.strip()),
        signal=SignalSpec(
            path=Path(values.signal_path.strip()),
            score_column=values.score_column.strip(),
            date_column=base.signal.date_column,
            ticker_column=base.signal.ticker_column,
            volatility_column=_optional_text(values.volatility_column),
            sector_column=_optional_text(values.sector_column),
            components=components,
        ),
        universe=UniverseSpec(
            excluded_tickers=_ticker_list(values.excluded_tickers),
            start_date=_optional_text(values.start_date),
            end_date=_optional_text(values.end_date),
        ),
        portfolio=PortfolioSpec(
            strategy_type=values.strategy_type,
            top_k=int(values.top_k),
            long_quantile=float(values.long_quantile),
            short_quantile=float(values.short_quantile),
            weighting=values.weighting,
            rebalance_frequency=values.rebalance_frequency,
        ),
        risk=RiskSpec(
            max_position_weight=(
                float(values.max_position_weight)
                if values.use_position_limit
                else None
            ),
            max_sector_weight=(
                float(values.max_sector_weight) if values.use_sector_limit else None
            ),
            min_dollar_volume=(
                float(values.min_dollar_volume)
                if values.use_liquidity_limit
                else None
            ),
            liquidity_window=int(values.liquidity_window),
        ),
        execution=ExecutionSpec(
            initial_capital=float(values.initial_capital),
            commission_bps=float(values.commission_bps),
            slippage_bps=float(values.slippage_bps),
            execution_lag_days=int(values.execution_lag_days),
            annual_cash_rate=float(values.annual_cash_rate),
            annual_borrow_rate=float(values.annual_borrow_rate),
        ),
        benchmark_ticker=_optional_text(values.benchmark_ticker),
        risk_free_rate=float(values.risk_free_rate),
    )
    spec.validate()
    return spec


def strategy_output_path(
    spec: StrategySpec,
    strategy_dir: str | Path = "strategies",
) -> Path:
    return Path(strategy_dir) / f"{spec.slug}.yaml"


def _ticker_list(value: str) -> tuple[str, ...]:
    normalized = value.replace("\n", ",")
    return tuple(
        dict.fromkeys(
            ticker.strip().upper()
            for ticker in normalized.split(",")
            if ticker.strip()
        )
    )


def _optional_text(value: str) -> str | None:
    normalized = value.strip()
    return normalized or None
