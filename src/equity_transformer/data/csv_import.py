from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_transformer.data.validation import (
    ADJUSTED_PRICE_STATUSES,
    MARKET_COLUMNS,
    validate_market_frame,
    validate_price_adjustment_status,
)


@dataclass(frozen=True)
class MarketCsvImportConfig:
    input_path: Path
    output_path: Path
    metadata_path: Path
    start_date: str | None = None
    end_date: str | None = None
    default_ticker: str | None = None
    price_adjustment_status: str = "unknown"
    require_adjusted_prices: bool = False


def load_market_csv_import_config(path: str | Path) -> MarketCsvImportConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}

    return MarketCsvImportConfig(
        input_path=Path(payload["input_path"]),
        output_path=Path(payload["output_path"]),
        metadata_path=Path(payload["metadata_path"]),
        start_date=(str(payload["start_date"]) if payload.get("start_date") else None),
        end_date=(str(payload["end_date"]) if payload.get("end_date") else None),
        default_ticker=(
            str(payload["default_ticker"]).upper()
            if payload.get("default_ticker")
            else None
        ),
        price_adjustment_status=str(
            payload.get("price_adjustment_status", "unknown")
        ),
        require_adjusted_prices=bool(
            payload.get("require_adjusted_prices", False)
        ),
    )


class MarketCsvImporter:
    _ALIASES = {
        "datetime": "date",
        "timestamp": "date",
        "symbol": "ticker",
        "tic": "ticker",
        "adjusted_close": "adj_close",
        "adjclose": "adj_close",
    }

    def __init__(self, config: MarketCsvImportConfig) -> None:
        self.config = config
        self._used_close_proxy = False
        validate_price_adjustment_status(config.price_adjustment_status)
        if (
            config.require_adjusted_prices
            and config.price_adjustment_status not in ADJUSTED_PRICE_STATUSES
        ):
            raise ValueError(
                "CSV workflow requires adjusted prices, but price adjustment "
                f"status is {config.price_adjustment_status}."
            )

    def run(self) -> pd.DataFrame:
        source_files = self._source_files()
        frames = [self._read_source(path) for path in source_files]
        if (
            self._used_close_proxy
            and self.config.price_adjustment_status in ADJUSTED_PRICE_STATUSES
        ):
            raise ValueError(
                "CSV declares adjusted prices but has no adj_close column."
            )
        panel = pd.concat(frames, ignore_index=True)
        panel = self._filter_dates(panel)
        panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
        validate_market_frame(panel)

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(self.config.output_path, index=False)
        self._write_manifest(panel, source_files)
        return panel

    def _source_files(self) -> list[Path]:
        source = self.config.input_path
        if source.is_file():
            if source.suffix.lower() != ".csv":
                raise ValueError(f"Input file must be CSV: {source}")
            return [source]
        if source.is_dir():
            files = sorted(source.glob("*.csv"))
            if files:
                return files
            raise ValueError(f"Input directory contains no CSV files: {source}")
        raise FileNotFoundError(f"CSV input path does not exist: {source}")

    def _read_source(self, path: Path) -> pd.DataFrame:
        native = pd.read_csv(path)
        if native.empty:
            raise ValueError(f"CSV file contains no rows: {path}")

        normalized = [self._normalize_column(column) for column in native.columns]
        duplicates = sorted({name for name in normalized if normalized.count(name) > 1})
        if duplicates:
            raise ValueError(f"CSV columns map to duplicate names {duplicates}: {path}")
        native.columns = normalized

        if "date" not in native.columns:
            raise ValueError(f"CSV file has no date column: {path}")
        if "ticker" not in native.columns:
            ticker = self.config.default_ticker
            if ticker is None and self.config.input_path.is_dir():
                ticker = path.stem.upper()
            if ticker is None:
                raise ValueError(
                    "Single-file CSV requires a ticker/symbol column or default_ticker."
                )
            native["ticker"] = ticker
        if "adj_close" not in native.columns and "close" in native.columns:
            native["adj_close"] = native["close"]
            self._used_close_proxy = True

        missing = sorted(set(MARKET_COLUMNS).difference(native.columns))
        if missing:
            raise ValueError(f"Missing required CSV columns {missing}: {path}")

        clean = native[MARKET_COLUMNS].copy()
        clean["date"] = pd.to_datetime(
            clean["date"], utc=True, errors="coerce"
        ).dt.tz_localize(None)
        clean["ticker"] = clean["ticker"].astype("string").str.strip().str.upper()
        numeric = ["open", "high", "low", "close", "adj_close", "volume"]
        clean[numeric] = clean[numeric].apply(pd.to_numeric, errors="coerce")
        invalid = clean[MARKET_COLUMNS].isna().any(axis=1)
        if invalid.any():
            rows = (invalid[invalid].index + 2).tolist()[:10]
            raise ValueError(f"Invalid required values in CSV rows {rows}: {path}")
        if (clean["volume"] % 1 != 0).any():
            raise ValueError(f"Volume must contain whole numbers: {path}")
        clean["volume"] = clean["volume"].astype("int64")
        return clean

    def _filter_dates(self, panel: pd.DataFrame) -> pd.DataFrame:
        filtered = panel
        if self.config.start_date:
            start = pd.Timestamp(self.config.start_date)
            filtered = filtered[filtered["date"] >= start]
        if self.config.end_date:
            filtered = filtered[filtered["date"] < pd.Timestamp(self.config.end_date)]
        if filtered.empty:
            raise ValueError("No CSV rows remain after date filtering.")
        return filtered.copy()

    @classmethod
    def _normalize_column(cls, column: object) -> str:
        name = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        while "__" in name:
            name = name.replace("__", "_")
        return cls._ALIASES.get(name, name)

    def _write_manifest(self, panel: pd.DataFrame, files: list[Path]) -> None:
        config = asdict(self.config)
        for key in ("input_path", "output_path", "metadata_path"):
            config[key] = str(config[key])
        manifest = {
            "run_utc": datetime.now(UTC).isoformat(),
            "source_type": "csv_import",
            "synthetic": False,
            "price_adjustment_status": self.config.price_adjustment_status,
            "require_adjusted_prices": self.config.require_adjusted_prices,
            "used_close_as_adjusted_close": self._used_close_proxy,
            "config": config,
            "sources": [
                {
                    "path": str(path),
                    "sha256": self._sha256(path),
                    "bytes": path.stat().st_size,
                }
                for path in files
            ],
            "rows": len(panel),
            "tickers": sorted(panel["ticker"].unique().tolist()),
            "min_date": panel["date"].min().isoformat(),
            "max_date": panel["date"].max().isoformat(),
        }
        self.config.metadata_path.write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
