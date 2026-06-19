from __future__ import annotations

import numpy as np
import pandas as pd

MARKET_COLUMNS = [
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
]

PRICE_ADJUSTMENT_STATUSES = {
    "provider_adjusted_ohlcv",
    "provider_adjusted_close",
    "unadjusted_close_proxy",
    "unknown",
}
ADJUSTED_PRICE_STATUSES = {
    "provider_adjusted_ohlcv",
    "provider_adjusted_close",
}


def validate_price_adjustment_status(status: str) -> None:
    if status not in PRICE_ADJUSTMENT_STATUSES:
        raise ValueError(f"Unknown price adjustment status: {status}")


def validate_market_frame(frame: pd.DataFrame) -> None:
    missing = set(MARKET_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required market columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Market data frame is empty.")
    if frame[["date", "ticker"]].duplicated().any():
        raise ValueError("Duplicate (date, ticker) observations found.")
    if frame["date"].isna().any() or frame["ticker"].isna().any():
        raise ValueError("Market data keys cannot be null.")

    numeric = ["open", "high", "low", "close", "adj_close", "volume"]
    if frame[numeric].isna().any().any():
        raise ValueError("Null values found in required market fields.")
    if not np.isfinite(frame[numeric].to_numpy(dtype="float64")).all():
        raise ValueError("Non-finite values found in required market fields.")
    if (frame[numeric] < 0).any().any():
        raise ValueError("Negative market values found.")

    price_max = frame[["open", "close", "low"]].max(axis=1)
    price_min = frame[["open", "close", "high"]].min(axis=1)
    if (frame["high"] < price_max).any():
        raise ValueError("High price is below open, close, or low.")
    if (frame["low"] > price_min).any():
        raise ValueError("Low price is above open, close, or high.")
