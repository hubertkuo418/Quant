from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from equity_transformer.data.config import DataConfig
from equity_transformer.data.providers import MarketDataProvider
from equity_transformer.data.universe import (
    filter_market_to_membership,
    load_universe_membership,
    membership_tickers,
)
from equity_transformer.data.validation import (
    ADJUSTED_PRICE_STATUSES,
    MARKET_COLUMNS,
    validate_market_frame,
    validate_price_adjustment_status,
)

LOGGER = logging.getLogger(__name__)


class MarketDataPipeline:
    def __init__(self, config: DataConfig, provider: MarketDataProvider) -> None:
        self.config = config
        self.provider = provider
        self.adjustment_status = provider.adjustment_status(config.auto_adjust)
        validate_price_adjustment_status(self.adjustment_status)
        if (
            config.require_adjusted_prices
            and self.adjustment_status not in ADJUSTED_PRICE_STATUSES
        ):
            raise ValueError(
                "Configured workflow requires corporate-action-adjusted prices, "
                f"but provider status is {self.adjustment_status}."
            )

    def run(
        self,
        tickers: tuple[str, ...] | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        membership = self._load_membership()
        configured = self.config.universe.tickers
        if membership is not None:
            configured = tuple(
                dict.fromkeys(
                    (*membership_tickers(membership), *configured)
                )
            )
        selected = tickers or tuple(
            dict.fromkeys(
                (*configured, *self.config.universe.always_include_tickers)
            )
        )
        start_date = start or self.config.start_date
        end_date = end or self.config.resolved_end_date
        frames: list[pd.DataFrame] = []
        failures: dict[str, str] = {}

        self.config.raw_dir.mkdir(parents=True, exist_ok=True)
        self.config.processed_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)

        for ticker in selected:
            LOGGER.info("Downloading %s", ticker)
            try:
                native = self._download_with_retries(ticker, start_date, end_date)
                clean = self._standardize(native, ticker)
                validate_market_frame(clean)
                clean.to_parquet(self.config.raw_dir / f"{ticker}.parquet", index=False)
                frames.append(clean)
            except Exception as exc:  # Continue so one symbol does not lose the run.
                failures[ticker] = f"{type(exc).__name__}: {exc}"
                LOGGER.exception("Failed to process %s", ticker)

        if not frames:
            self._write_manifest(selected, start_date, end_date, failures, None)
            raise RuntimeError("No ticker data was downloaded successfully.")

        success_rate = len(frames) / len(selected)
        if success_rate < self.config.minimum_success_rate:
            self._write_manifest(selected, start_date, end_date, failures, None)
            raise RuntimeError(
                f"Market download success rate {success_rate:.1%} is below "
                f"required {self.config.minimum_success_rate:.1%}; canonical "
                "panel was not replaced."
            )

        panel = (
            pd.concat(frames, ignore_index=True)
            .sort_values(["date", "ticker"])
            .reset_index(drop=True)
        )
        if membership is not None:
            panel = filter_market_to_membership(
                panel,
                membership,
                self.config.universe.always_include_tickers,
            )
            if panel.empty:
                raise RuntimeError(
                    "Point-in-time universe filtering removed all market rows."
                )
        validate_market_frame(panel)
        panel.to_parquet(self.config.processed_path, index=False)
        self._write_manifest(selected, start_date, end_date, failures, panel)
        return panel

    def _load_membership(self) -> pd.DataFrame | None:
        path = self.config.universe.membership_path
        return load_universe_membership(path) if path is not None else None

    def _download_with_retries(
        self, ticker: str, start_date: str, end_date: str
    ) -> pd.DataFrame:
        attempts = self.config.max_retries + 1
        for attempt in range(1, attempts + 1):
            frame = self.provider.download(
                ticker=ticker,
                start=start_date,
                end=end_date,
                interval=self.config.interval,
                auto_adjust=self.config.auto_adjust,
            )
            if not frame.empty:
                return frame
            if attempt < attempts:
                delay = self.config.retry_delay_seconds * (2 ** (attempt - 1))
                LOGGER.warning(
                    "%s returned no rows; retrying in %.1f seconds (%d/%d)",
                    ticker,
                    delay,
                    attempt,
                    attempts,
                )
                time.sleep(delay)
        raise ValueError(f"Provider returned no rows after {attempts} attempts.")

    @staticmethod
    def _standardize(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
        if frame.empty:
            raise ValueError("Provider returned no rows.")

        native = frame.copy()
        if isinstance(native.columns, pd.MultiIndex):
            native.columns = native.columns.get_level_values(0)

        native = native.reset_index()
        native.columns = [
            str(column).strip().lower().replace(" ", "_") for column in native.columns
        ]
        date_column = "date" if "date" in native.columns else "datetime"
        if date_column not in native.columns:
            raise ValueError("Provider frame has no Date or Datetime column.")

        if "adj_close" not in native.columns:
            native["adj_close"] = native["close"]

        clean = native.rename(columns={date_column: "date"})
        clean["date"] = pd.to_datetime(clean["date"], utc=True).dt.tz_localize(None)
        clean["ticker"] = ticker.upper()
        clean = clean[MARKET_COLUMNS].copy()

        numeric = ["open", "high", "low", "close", "adj_close", "volume"]
        clean[numeric] = clean[numeric].apply(pd.to_numeric, errors="coerce")
        clean = clean.dropna(subset=MARKET_COLUMNS)
        clean["volume"] = clean["volume"].astype("int64")
        return clean.sort_values("date").drop_duplicates(["date", "ticker"])

    def _write_manifest(
        self,
        tickers: tuple[str, ...],
        start: str,
        end: str,
        failures: dict[str, str],
        panel: pd.DataFrame | None,
    ) -> None:
        run_time = datetime.now(UTC)
        config_payload = asdict(self.config)
        config_payload["raw_dir"] = str(self.config.raw_dir)
        config_payload["processed_path"] = str(self.config.processed_path)
        config_payload["metadata_dir"] = str(self.config.metadata_dir)
        config_payload["universe"]["tickers"] = list(
            config_payload["universe"]["tickers"]
        )
        config_payload["universe"]["membership_path"] = (
            str(self.config.universe.membership_path)
            if self.config.universe.membership_path is not None
            else None
        )
        config_payload["universe"]["always_include_tickers"] = list(
            config_payload["universe"]["always_include_tickers"]
        )
        manifest: dict[str, Any] = {
            "run_utc": run_time.isoformat(),
            "provider": self.config.provider,
            "adjusted_close_policy": self.adjustment_status,
            "price_adjustment_status": self.adjustment_status,
            "require_adjusted_prices": self.config.require_adjusted_prices,
            "universe_membership_policy": (
                "point_in_time_intervals"
                if self.config.universe.membership_path is not None
                else "static_ticker_list"
            ),
            "requested_tickers": list(tickers),
            "start_date": start,
            "end_date_exclusive": end,
            "failures": failures,
            "config": config_payload,
            "rows": 0 if panel is None else len(panel),
            "successful_tickers": (
                [] if panel is None else sorted(panel["ticker"].unique().tolist())
            ),
            "min_date": None if panel is None else panel["date"].min().isoformat(),
            "max_date": None if panel is None else panel["date"].max().isoformat(),
        }
        path = self.config.metadata_dir / f"market_{run_time:%Y%m%dT%H%M%SZ}.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
