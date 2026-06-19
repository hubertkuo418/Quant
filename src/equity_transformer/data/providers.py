from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import timedelta
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf


class MarketDataProvider(ABC):
    def adjustment_status(self, auto_adjust: bool) -> str:
        return "unknown"

    @abstractmethod
    def download(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        """Download one ticker and return the provider-native frame."""


class YahooFinanceProvider(MarketDataProvider):
    def adjustment_status(self, auto_adjust: bool) -> str:
        return "provider_adjusted_ohlcv" if auto_adjust else "provider_adjusted_close"

    def download(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        return yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            actions=False,
            progress=False,
            threads=False,
            timeout=30,
        )


class NasdaqProvider(MarketDataProvider):
    endpoint = "https://api.nasdaq.com/api/quote/{ticker}/historical"
    symbol_aliases = {"BRK-B": "BRK.B"}
    etf_tickers = {"SPY"}

    def adjustment_status(self, auto_adjust: bool) -> str:
        return "unadjusted_close_proxy"

    def download(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("Nasdaq provider currently supports only daily data.")
        if auto_adjust:
            raise ValueError("Nasdaq provider does not supply adjusted OHLC data.")
        end_inclusive = pd.Timestamp(end) - timedelta(days=1)
        normalized_ticker = ticker.upper()
        provider_ticker = self.symbol_aliases.get(normalized_ticker, normalized_ticker)
        query = urlencode(
            {
                "assetclass": (
                    "etf" if normalized_ticker in self.etf_tickers else "stocks"
                ),
                "fromdate": pd.Timestamp(start).date().isoformat(),
                "todate": end_inclusive.date().isoformat(),
                "limit": 5000,
            }
        )
        payload = self._request_json(
            f"{self.endpoint.format(ticker=provider_ticker)}?{query}"
        )
        data = payload.get("data") or {}
        table = data.get("tradesTable") or {}
        rows = table.get("rows") or []
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            }
        )
        for column in ["Open", "High", "Low", "Close", "Volume"]:
            frame[column] = pd.to_numeric(
                frame[column].astype(str).str.replace(r"[$,]", "", regex=True),
                errors="coerce",
            )
        frame["Adj Close"] = frame["Close"]
        frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
        return frame.set_index("Date").sort_index()

    @staticmethod
    def _request_json(url: str) -> dict[str, object]:
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json, text/plain, */*",
            },
        )
        with urlopen(request, timeout=45) as response:  # noqa: S310
            return json.loads(response.read().decode("utf-8"))


def create_provider(name: str) -> MarketDataProvider:
    if name.lower() == "yahoo":
        return YahooFinanceProvider()
    if name.lower() == "nasdaq":
        return NasdaqProvider()
    raise ValueError(f"Unsupported market data provider: {name}")
