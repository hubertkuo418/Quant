from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class BacktestConfig:
    market_path: Path
    weights_path: Path
    output_dir: Path
    initial_capital: float
    commission_bps: float
    slippage_bps: float
    annualization_factor: int
    risk_free_rate: float
    execution_lag_days: int = 0
    min_dollar_volume: float | None = None
    liquidity_window: int = 20
    benchmark_ticker: str | None = None
    annual_cash_rate: float = 0.0
    annual_borrow_rate: float = 0.0

    @property
    def transaction_cost_rate(self) -> float:
        return (self.commission_bps + self.slippage_bps) / 10_000


def load_backtest_config(path: str | Path) -> BacktestConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return BacktestConfig(
        market_path=Path(payload["market_path"]),
        weights_path=Path(payload["weights_path"]),
        output_dir=Path(payload["output_dir"]),
        initial_capital=float(payload["initial_capital"]),
        commission_bps=float(payload["commission_bps"]),
        slippage_bps=float(payload["slippage_bps"]),
        annualization_factor=int(payload["annualization_factor"]),
        risk_free_rate=float(payload["risk_free_rate"]),
        execution_lag_days=int(payload.get("execution_lag_days", 0)),
        min_dollar_volume=(
            float(payload["min_dollar_volume"])
            if payload.get("min_dollar_volume") is not None
            else None
        ),
        liquidity_window=int(payload.get("liquidity_window", 20)),
        benchmark_ticker=(
            str(payload["benchmark_ticker"])
            if payload.get("benchmark_ticker") is not None
            else None
        ),
        annual_cash_rate=float(payload.get("annual_cash_rate", 0.0)),
        annual_borrow_rate=float(payload.get("annual_borrow_rate", 0.0)),
    )
