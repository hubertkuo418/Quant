from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StrategyConfig:
    signal_path: Path
    output_path: Path
    metadata_path: Path
    date_column: str
    ticker_column: str
    score_column: str
    strategy_type: str
    top_k: int
    long_quantile: float
    short_quantile: float
    weighting: str
    rebalance_frequency: str
    volatility_column: str | None = None
    sector_column: str | None = None
    max_sector_weight: float | None = None
    max_position_weight: float | None = None
    excluded_tickers: tuple[str, ...] = ()
    start_date: str | None = None
    end_date: str | None = None


def load_strategy_config(path: str | Path) -> StrategyConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return StrategyConfig(
        signal_path=Path(payload["signal_path"]),
        output_path=Path(payload["output_path"]),
        metadata_path=Path(payload["metadata_path"]),
        date_column=str(payload["date_column"]),
        ticker_column=str(payload["ticker_column"]),
        score_column=str(payload["score_column"]),
        strategy_type=str(payload["strategy_type"]),
        top_k=int(payload["top_k"]),
        long_quantile=float(payload["long_quantile"]),
        short_quantile=float(payload["short_quantile"]),
        weighting=str(payload["weighting"]),
        rebalance_frequency=str(payload["rebalance_frequency"]),
        volatility_column=(
            str(payload["volatility_column"])
            if payload.get("volatility_column") is not None
            else None
        ),
        sector_column=(
            str(payload["sector_column"])
            if payload.get("sector_column") is not None
            else None
        ),
        max_sector_weight=(
            float(payload["max_sector_weight"])
            if payload.get("max_sector_weight") is not None
            else None
        ),
        max_position_weight=(
            float(payload["max_position_weight"])
            if payload.get("max_position_weight") is not None
            else None
        ),
        excluded_tickers=tuple(
            str(ticker).upper() for ticker in payload.get("excluded_tickers", [])
        ),
        start_date=str(payload["start_date"]) if payload.get("start_date") else None,
        end_date=str(payload["end_date"]) if payload.get("end_date") else None,
    )
