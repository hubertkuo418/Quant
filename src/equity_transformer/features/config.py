from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FeatureConfig:
    market_path: Path
    output_path: Path
    metadata_dir: Path
    fundamentals_path: Path | None
    news_path: Path | None
    return_windows: tuple[int, ...]
    volatility_window: int
    rsi_window: int
    moving_average_windows: tuple[int, ...]
    volume_window: int
    drop_incomplete: bool
    momentum_windows: tuple[int, ...] = ()
    volatility_windows: tuple[int, ...] = ()
    drawdown_windows: tuple[int, ...] = ()
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    atr_window: int = 14
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    news_market_timezone: str = "America/New_York"
    news_cutoff_time: str = "09:30"


def load_feature_config(path: str | Path) -> FeatureConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)

    return FeatureConfig(
        market_path=Path(payload["market_path"]),
        output_path=Path(payload["output_path"]),
        metadata_dir=Path(payload["metadata_dir"]),
        fundamentals_path=_optional_path(payload.get("fundamentals_path")),
        news_path=_optional_path(payload.get("news_path")),
        return_windows=tuple(int(value) for value in payload["return_windows"]),
        volatility_window=int(payload["volatility_window"]),
        rsi_window=int(payload["rsi_window"]),
        moving_average_windows=tuple(
            int(value) for value in payload["moving_average_windows"]
        ),
        volume_window=int(payload["volume_window"]),
        drop_incomplete=bool(payload.get("drop_incomplete", False)),
        momentum_windows=tuple(
            int(value) for value in payload.get("momentum_windows", [])
        ),
        volatility_windows=tuple(
            int(value) for value in payload.get("volatility_windows", [])
        ),
        drawdown_windows=tuple(
            int(value) for value in payload.get("drawdown_windows", [])
        ),
        macd_fast=int(payload.get("macd_fast", 12)),
        macd_slow=int(payload.get("macd_slow", 26)),
        macd_signal=int(payload.get("macd_signal", 9)),
        atr_window=int(payload.get("atr_window", 14)),
        bollinger_window=int(payload.get("bollinger_window", 20)),
        bollinger_std=float(payload.get("bollinger_std", 2.0)),
        news_market_timezone=str(
            payload.get("news_market_timezone", "America/New_York")
        ),
        news_cutoff_time=str(payload.get("news_cutoff_time", "09:30")),
    )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None
