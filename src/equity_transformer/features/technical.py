from __future__ import annotations

import numpy as np
import pandas as pd

from equity_transformer.features.config import FeatureConfig


def build_technical_features(
    market: pd.DataFrame, config: FeatureConfig
) -> pd.DataFrame:
    frame = market.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    grouped = frame.groupby("ticker", sort=False, group_keys=False)

    for window in config.return_windows:
        frame[f"return_{window}d"] = grouped["adj_close"].pct_change(
            periods=window, fill_method=None
        )
    for window in config.momentum_windows:
        frame[f"momentum_{window}d"] = grouped["adj_close"].pct_change(
            periods=window, fill_method=None
        )

    daily_return = grouped["adj_close"].pct_change(fill_method=None)
    frame[f"volatility_{config.volatility_window}d"] = (
        daily_return.groupby(frame["ticker"])
        .rolling(config.volatility_window, min_periods=config.volatility_window)
        .std()
        .reset_index(level=0, drop=True)
    )
    for window in config.volatility_windows:
        frame[f"volatility_{window}d"] = (
            daily_return.groupby(frame["ticker"])
            .rolling(window, min_periods=window)
            .std()
            .reset_index(level=0, drop=True)
        )
    for window in config.drawdown_windows:
        frame[f"max_drawdown_{window}d"] = grouped["adj_close"].transform(
            lambda prices, size=window: _rolling_max_drawdown(prices, size)
        )
    frame[f"rsi_{config.rsi_window}d"] = grouped["adj_close"].transform(
        lambda prices: _rsi(prices, config.rsi_window)
    )
    frame = _add_macd(frame, config)
    frame = _add_atr(frame, config)
    frame = _add_bollinger(frame, config)

    for window in config.moving_average_windows:
        moving_average = grouped["adj_close"].transform(
            lambda prices, size=window: prices.rolling(
                size, min_periods=size
            ).mean()
        )
        frame[f"price_ma_{window}d_ratio"] = frame["adj_close"] / moving_average - 1

    log_volume = np.log1p(frame["volume"])
    volume_mean = (
        log_volume.groupby(frame["ticker"])
        .rolling(config.volume_window, min_periods=config.volume_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    volume_std = (
        log_volume.groupby(frame["ticker"])
        .rolling(config.volume_window, min_periods=config.volume_window)
        .std()
        .reset_index(level=0, drop=True)
    )
    frame[f"log_volume_zscore_{config.volume_window}d"] = (
        log_volume - volume_mean
    ) / volume_std.replace(0, np.nan)

    dollar_volume = frame["adj_close"] * frame["volume"]
    frame[f"avg_dollar_volume_{config.volume_window}d"] = (
        dollar_volume.groupby(frame["ticker"])
        .rolling(config.volume_window, min_periods=config.volume_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return frame.sort_values(["date", "ticker"]).reset_index(drop=True)


def _rsi(prices: pd.Series, window: int) -> pd.Series:
    change = prices.diff()
    average_gain = change.clip(lower=0).rolling(window, min_periods=window).mean()
    average_loss = -change.clip(upper=0).rolling(window, min_periods=window).mean()
    denominator = average_gain + average_loss
    rsi = 100 * average_gain / denominator
    return rsi.where(denominator != 0, 50.0)


def _rolling_max_drawdown(prices: pd.Series, window: int) -> pd.Series:
    return prices.rolling(window, min_periods=window).apply(
        lambda values: float((values / np.maximum.accumulate(values) - 1).min()),
        raw=True,
    )


def _add_macd(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    grouped = frame.groupby("ticker", sort=False, group_keys=False)
    fast = grouped["adj_close"].transform(
        lambda prices: prices.ewm(span=config.macd_fast, adjust=False).mean()
    )
    slow = grouped["adj_close"].transform(
        lambda prices: prices.ewm(span=config.macd_slow, adjust=False).mean()
    )
    macd = fast - slow
    signal = macd.groupby(frame["ticker"]).transform(
        lambda values: values.ewm(span=config.macd_signal, adjust=False).mean()
    )
    frame["macd"] = macd
    frame["macd_signal"] = signal
    frame["macd_histogram"] = macd - signal
    return frame


def _add_atr(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    previous_close = frame.groupby("ticker", sort=False)["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - previous_close).abs(),
            (frame["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame[f"atr_{config.atr_window}d"] = (
        true_range.groupby(frame["ticker"])
        .rolling(config.atr_window, min_periods=config.atr_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    frame[f"atr_{config.atr_window}d_ratio"] = (
        frame[f"atr_{config.atr_window}d"] / frame["adj_close"]
    )
    return frame


def _add_bollinger(frame: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    grouped = frame.groupby("ticker", sort=False, group_keys=False)
    mean = grouped["adj_close"].transform(
        lambda prices: prices.rolling(
            config.bollinger_window, min_periods=config.bollinger_window
        ).mean()
    )
    std = grouped["adj_close"].transform(
        lambda prices: prices.rolling(
            config.bollinger_window, min_periods=config.bollinger_window
        ).std()
    )
    upper = mean + config.bollinger_std * std
    lower = mean - config.bollinger_std * std
    denominator = (upper - lower).replace(0, np.nan)
    frame[f"bollinger_percent_b_{config.bollinger_window}d"] = (
        frame["adj_close"] - lower
    ) / denominator
    frame[f"bollinger_bandwidth_{config.bollinger_window}d"] = denominator / mean
    return frame
