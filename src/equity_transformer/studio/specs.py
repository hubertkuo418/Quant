from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class SignalComponent:
    column: str
    weight: float = 1.0
    transform: str = "cross_sectional_zscore"


@dataclass(frozen=True)
class SignalSpec:
    path: Path
    score_column: str
    date_column: str = "date"
    ticker_column: str = "ticker"
    volatility_column: str | None = None
    sector_column: str | None = None
    components: tuple[SignalComponent, ...] = ()


@dataclass(frozen=True)
class UniverseSpec:
    excluded_tickers: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None


@dataclass(frozen=True)
class PortfolioSpec:
    strategy_type: str = "long_only_top_k"
    top_k: int = 10
    long_quantile: float = 0.2
    short_quantile: float = 0.2
    weighting: str = "equal"
    rebalance_frequency: str = "W-FRI"


@dataclass(frozen=True)
class RiskSpec:
    max_position_weight: float | None = None
    max_sector_weight: float | None = None
    min_dollar_volume: float | None = None
    liquidity_window: int = 20


@dataclass(frozen=True)
class ExecutionSpec:
    initial_capital: float = 1_000_000.0
    commission_bps: float = 5.0
    slippage_bps: float = 5.0
    execution_lag_days: int = 1
    annual_cash_rate: float = 0.0
    annual_borrow_rate: float = 0.0


@dataclass(frozen=True)
class StrategySpec:
    name: str
    version: str
    description: str
    market_path: Path
    signal: SignalSpec
    universe: UniverseSpec = UniverseSpec()
    portfolio: PortfolioSpec = PortfolioSpec()
    risk: RiskSpec = RiskSpec()
    execution: ExecutionSpec = ExecutionSpec()
    benchmark_ticker: str | None = "SPY"
    annualization_factor: int = 252
    risk_free_rate: float = 0.0
    parent_run_id: str | None = None

    @property
    def slug(self) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", self.name.lower()).strip("-")
        return slug or "strategy"

    @property
    def spec_hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return _stringify_paths(asdict(self))

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("Strategy name cannot be empty.")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", self.version):
            raise ValueError("Strategy version must be a safe identifier.")
        if self.portfolio.strategy_type not in {
            "long_only_top_k",
            "long_short_quantile",
        }:
            raise ValueError("Unsupported portfolio strategy_type.")
        if self.portfolio.weighting not in {
            "equal",
            "score",
            "inverse_volatility",
            "risk_parity",
        }:
            raise ValueError("Unsupported portfolio weighting.")
        if self.portfolio.top_k <= 0:
            raise ValueError("portfolio.top_k must be positive.")
        for name in ("long_quantile", "short_quantile"):
            value = getattr(self.portfolio, name)
            if not 0 < value <= 0.5:
                raise ValueError(f"portfolio.{name} must be in (0, 0.5].")
        if self.portfolio.weighting in {"inverse_volatility", "risk_parity"}:
            if self.signal.volatility_column is None:
                raise ValueError(
                    f"{self.portfolio.weighting} requires signal.volatility_column."
                )
        component_columns = [item.column for item in self.signal.components]
        if len(component_columns) != len(set(component_columns)):
            raise ValueError("Signal component columns must be unique.")
        for component in self.signal.components:
            if component.transform not in {
                "raw",
                "cross_sectional_rank",
                "cross_sectional_zscore",
            }:
                raise ValueError(f"Unsupported signal transform: {component.transform}")
            if component.weight == 0:
                raise ValueError("Signal component weights cannot be zero.")
        for name in ("max_position_weight", "max_sector_weight"):
            value = getattr(self.risk, name)
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"risk.{name} must be in (0, 1].")
        if (
            self.risk.max_sector_weight is not None
            and self.signal.sector_column is None
        ):
            raise ValueError("max_sector_weight requires signal.sector_column.")
        if self.risk.liquidity_window <= 0:
            raise ValueError("risk.liquidity_window must be positive.")
        if self.execution.initial_capital <= 0:
            raise ValueError("execution.initial_capital must be positive.")
        if min(self.execution.commission_bps, self.execution.slippage_bps) < 0:
            raise ValueError("Execution costs cannot be negative.")
        if self.execution.execution_lag_days < 0:
            raise ValueError("execution.execution_lag_days cannot be negative.")
        if self.annualization_factor <= 0:
            raise ValueError("annualization_factor must be positive.")


def load_strategy_spec(path: str | Path) -> StrategySpec:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    spec = strategy_spec_from_dict(payload)
    spec.validate()
    return spec


def save_strategy_spec(spec: StrategySpec, path: str | Path) -> None:
    spec.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(yaml.safe_dump(spec.to_dict(), sort_keys=False), encoding="utf-8")


def strategy_spec_from_dict(payload: dict[str, Any]) -> StrategySpec:
    signal = payload["signal"]
    universe = payload.get("universe", {})
    portfolio = payload.get("portfolio", {})
    risk = payload.get("risk", {})
    execution = payload.get("execution", {})
    return StrategySpec(
        name=str(payload["name"]),
        version=str(payload.get("version", "1.0.0")),
        description=str(payload.get("description", "")),
        market_path=Path(payload["market_path"]),
        signal=SignalSpec(
            path=Path(signal["path"]),
            score_column=str(signal["score_column"]),
            date_column=str(signal.get("date_column", "date")),
            ticker_column=str(signal.get("ticker_column", "ticker")),
            volatility_column=_optional_string(signal.get("volatility_column")),
            sector_column=_optional_string(signal.get("sector_column")),
            components=tuple(
                SignalComponent(
                    column=str(item["column"]),
                    weight=float(item.get("weight", 1.0)),
                    transform=str(
                        item.get("transform", "cross_sectional_zscore")
                    ),
                )
                for item in signal.get("components", [])
            ),
        ),
        universe=UniverseSpec(
            excluded_tickers=tuple(
                str(value).upper() for value in universe.get("excluded_tickers", [])
            ),
            start_date=_optional_string(universe.get("start_date")),
            end_date=_optional_string(universe.get("end_date")),
        ),
        portfolio=PortfolioSpec(
            strategy_type=str(portfolio.get("strategy_type", "long_only_top_k")),
            top_k=int(portfolio.get("top_k", 10)),
            long_quantile=float(portfolio.get("long_quantile", 0.2)),
            short_quantile=float(portfolio.get("short_quantile", 0.2)),
            weighting=str(portfolio.get("weighting", "equal")),
            rebalance_frequency=str(portfolio.get("rebalance_frequency", "W-FRI")),
        ),
        risk=RiskSpec(
            max_position_weight=_optional_float(risk.get("max_position_weight")),
            max_sector_weight=_optional_float(risk.get("max_sector_weight")),
            min_dollar_volume=_optional_float(risk.get("min_dollar_volume")),
            liquidity_window=int(risk.get("liquidity_window", 20)),
        ),
        execution=ExecutionSpec(
            initial_capital=float(execution.get("initial_capital", 1_000_000)),
            commission_bps=float(execution.get("commission_bps", 5)),
            slippage_bps=float(execution.get("slippage_bps", 5)),
            execution_lag_days=int(execution.get("execution_lag_days", 1)),
            annual_cash_rate=float(execution.get("annual_cash_rate", 0)),
            annual_borrow_rate=float(execution.get("annual_borrow_rate", 0)),
        ),
        benchmark_ticker=_optional_string(payload.get("benchmark_ticker")),
        annualization_factor=int(payload.get("annualization_factor", 252)),
        risk_free_rate=float(payload.get("risk_free_rate", 0)),
        parent_run_id=_optional_string(payload.get("parent_run_id")),
    )


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None and str(value).strip() else None


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _stringify_paths(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_stringify_paths(item) for item in value]
    return value
