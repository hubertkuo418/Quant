from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class UniverseConfig:
    name: str
    tickers: tuple[str, ...]


@dataclass(frozen=True)
class DataConfig:
    provider: str
    start_date: str
    end_date: str | None
    interval: str
    auto_adjust: bool
    max_retries: int
    retry_delay_seconds: float
    raw_dir: Path
    processed_path: Path
    metadata_dir: Path
    universe: UniverseConfig
    minimum_success_rate: float = 1.0

    @property
    def resolved_end_date(self) -> str:
        return self.end_date or date.today().isoformat()


def load_data_config(path: str | Path) -> DataConfig:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)

    universe = payload["universe"]
    tickers = tuple(dict.fromkeys(ticker.upper() for ticker in universe["tickers"]))
    if not tickers:
        raise ValueError("Universe must contain at least one ticker.")

    minimum_success_rate = float(payload.get("minimum_success_rate", 1.0))
    if not 0 < minimum_success_rate <= 1:
        raise ValueError("minimum_success_rate must be in (0, 1].")

    return DataConfig(
        provider=payload["provider"],
        start_date=str(payload["start_date"]),
        end_date=str(payload["end_date"]) if payload.get("end_date") else None,
        interval=payload.get("interval", "1d"),
        auto_adjust=bool(payload.get("auto_adjust", False)),
        max_retries=int(payload.get("max_retries", 3)),
        retry_delay_seconds=float(payload.get("retry_delay_seconds", 2.0)),
        raw_dir=Path(payload["raw_dir"]),
        processed_path=Path(payload["processed_path"]),
        metadata_dir=Path(payload["metadata_dir"]),
        universe=UniverseConfig(name=universe["name"], tickers=tickers),
        minimum_success_rate=minimum_success_rate,
    )
