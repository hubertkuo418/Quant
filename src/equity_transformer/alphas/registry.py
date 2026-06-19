from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

AlphaFunction = Callable[[pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class AlphaDefinition:
    name: str
    family: str
    lookback: int
    description: str
    function: AlphaFunction


def alpha_registry() -> dict[str, AlphaDefinition]:
    return {
        definition.name: definition
        for definition in [
            AlphaDefinition(
                name="alpha_momentum_20d",
                family="momentum",
                lookback=20,
                description="Trailing 20-day adjusted-close return.",
                function=lambda frame: _grouped_pct_change(frame, 20),
            ),
            AlphaDefinition(
                name="alpha_reversal_5d",
                family="reversal",
                lookback=5,
                description="Negative trailing 5-day adjusted-close return.",
                function=lambda frame: -_grouped_pct_change(frame, 5),
            ),
            AlphaDefinition(
                name="alpha_volume_price_corr_20d",
                family="price_volume",
                lookback=20,
                description="20-day rolling correlation between price and volume.",
                function=lambda frame: _rolling_corr(frame, 20),
            ),
            AlphaDefinition(
                name="alpha_volatility_reversal_20d",
                family="volatility",
                lookback=20,
                description="Negative 20-day realized volatility.",
                function=lambda frame: -_rolling_volatility(frame, 20),
            ),
            AlphaDefinition(
                name="alpha_price_position_20d",
                family="technical",
                lookback=20,
                description="Close location inside trailing 20-day high-low range.",
                function=lambda frame: _price_position(frame, 20),
            ),
            AlphaDefinition(
                name="alpha101_001",
                family="alpha101",
                lookback=20,
                description="Cross-sectional rank of 20-day momentum.",
                function=lambda frame: _cross_sectional_rank(
                    _grouped_pct_change(frame, 20), frame
                ),
            ),
            AlphaDefinition(
                name="alpha101_002",
                family="alpha101",
                lookback=6,
                description="Negative rank of 2-day price-volume correlation.",
                function=lambda frame: -_cross_sectional_rank(
                    _rolling_corr(frame, 6), frame
                ),
            ),
            AlphaDefinition(
                name="alpha101_003",
                family="alpha101",
                lookback=10,
                description="Ranked 10-day close-to-open reversal.",
                function=lambda frame: _cross_sectional_rank(
                    _grouped_delta(frame["open"] - frame["close"], frame, 10), frame
                ),
            ),
            AlphaDefinition(
                name="alpha101_004",
                family="alpha101",
                lookback=9,
                description="Negative rank of 9-day low-price position.",
                function=lambda frame: -_cross_sectional_rank(
                    _price_position(frame, 9), frame
                ),
            ),
            AlphaDefinition(
                name="alpha101_005",
                family="alpha101",
                lookback=10,
                description="Rank of open relative to trailing VWAP proxy.",
                function=lambda frame: _cross_sectional_rank(
                    frame["open"] / _rolling_vwap_proxy(frame, 10) - 1, frame
                ),
            ),
            AlphaDefinition(
                name="alpha101_006",
                family="alpha101",
                lookback=10,
                description="Negative price-volume correlation.",
                function=lambda frame: -_rolling_corr(frame, 10),
            ),
            AlphaDefinition(
                name="alpha101_007",
                family="alpha101",
                lookback=20,
                description="Volume-conditioned short-term reversal.",
                function=lambda frame: _volume_conditioned_reversal(frame, 20, 7),
            ),
            AlphaDefinition(
                name="alpha101_008",
                family="alpha101",
                lookback=5,
                description="Rank of five-day high-low range expansion.",
                function=lambda frame: _cross_sectional_rank(
                    _grouped_delta(frame["high"] - frame["low"], frame, 5), frame
                ),
            ),
            AlphaDefinition(
                name="alpha101_009",
                family="alpha101",
                lookback=5,
                description="Signed five-day close delta.",
                function=lambda frame: _signed_power(
                    _grouped_delta(frame["close"], frame, 5), 1
                ),
            ),
            AlphaDefinition(
                name="alpha101_010",
                family="alpha101",
                lookback=20,
                description="Negative rank of 20-day volatility.",
                function=lambda frame: -_cross_sectional_rank(
                    _rolling_volatility(frame, 20), frame
                ),
            ),
            AlphaDefinition(
                name="alpha101_011",
                family="alpha101",
                lookback=10,
                description="Ranked close deviation from a trailing VWAP proxy.",
                function=lambda frame: _cross_sectional_rank(
                    frame["close"] / _rolling_vwap_proxy(frame, 10) - 1, frame
                ),
            ),
            AlphaDefinition(
                name="alpha101_012",
                family="alpha101",
                lookback=1,
                description="Price reversal signed by one-day volume direction.",
                function=lambda frame: -_sign(
                    _grouped_delta(frame["volume"], frame, 1)
                )
                * _grouped_delta(frame["close"], frame, 1),
            ),
            AlphaDefinition(
                name="alpha101_013",
                family="alpha101",
                lookback=5,
                description="Negative rank of five-day close-volume covariance.",
                function=lambda frame: -_cross_sectional_rank(
                    _rolling_covariance(
                        _cross_sectional_rank(frame["close"], frame),
                        _cross_sectional_rank(frame["volume"], frame),
                        frame,
                        5,
                    ),
                    frame,
                ),
            ),
            AlphaDefinition(
                name="alpha101_014",
                family="alpha101",
                lookback=10,
                description="Return rank scaled by open-volume correlation.",
                function=lambda frame: -_cross_sectional_rank(
                    _grouped_pct_change(frame, 1), frame
                )
                * _rolling_pair_corr(frame["open"], frame["volume"], frame, 10),
            ),
            AlphaDefinition(
                name="alpha101_015",
                family="alpha101",
                lookback=5,
                description="Negative sum of ranked high-volume correlation.",
                function=lambda frame: -_rolling_sum(
                    _cross_sectional_rank(
                        _rolling_pair_corr(
                            frame["high"], frame["volume"], frame, 3
                        ),
                        frame,
                    ),
                    frame,
                    3,
                ),
            ),
            AlphaDefinition(
                name="alpha101_016",
                family="alpha101",
                lookback=5,
                description="Negative rank of high-volume covariance.",
                function=lambda frame: -_cross_sectional_rank(
                    _rolling_covariance(
                        _cross_sectional_rank(frame["high"], frame),
                        _cross_sectional_rank(frame["volume"], frame),
                        frame,
                        5,
                    ),
                    frame,
                ),
            ),
            AlphaDefinition(
                name="alpha101_017",
                family="alpha101",
                lookback=20,
                description="Price-position, acceleration, and volume interaction.",
                function=lambda frame: -_cross_sectional_rank(
                    _price_position(frame, 10), frame
                )
                * _cross_sectional_rank(
                    _grouped_delta(frame["close"], frame, 2), frame
                )
                * _cross_sectional_rank(
                    frame["volume"] / _rolling_mean(frame["volume"], frame, 20),
                    frame,
                ),
            ),
            AlphaDefinition(
                name="alpha101_018",
                family="alpha101",
                lookback=10,
                description="Negative rank of intraday spread and price correlation.",
                function=lambda frame: -_cross_sectional_rank(
                    _rolling_std(frame["close"] - frame["open"], frame, 5)
                    + (frame["close"] - frame["open"]).abs()
                    + _rolling_pair_corr(
                        frame["close"], frame["open"], frame, 10
                    ),
                    frame,
                ),
            ),
            AlphaDefinition(
                name="alpha101_019",
                family="alpha101",
                lookback=60,
                description="Seven-day direction reversal scaled by long momentum.",
                function=lambda frame: -_sign(
                    _grouped_delta(frame["close"], frame, 7)
                )
                * (
                    1
                    + _cross_sectional_rank(
                        _grouped_pct_change(frame, 60), frame
                    )
                ),
            ),
            AlphaDefinition(
                name="alpha101_020",
                family="alpha101",
                lookback=1,
                description="Negative product of ranked overnight OHLC gaps.",
                function=lambda frame: -_overnight_gap_rank_product(frame),
            ),
        ]
    }


def _grouped_pct_change(frame: pd.DataFrame, window: int) -> pd.Series:
    return frame.groupby("ticker", sort=False)["adj_close"].pct_change(
        window, fill_method=None
    )


def _grouped_delta(values: pd.Series, frame: pd.DataFrame, window: int) -> pd.Series:
    return values.groupby(frame["ticker"], sort=False).diff(window)


def _cross_sectional_rank(values: pd.Series, frame: pd.DataFrame) -> pd.Series:
    return values.groupby(frame["date"]).rank(pct=True)


def _signed_power(values: pd.Series, exponent: float) -> pd.Series:
    return values.abs().pow(exponent) * values.where(values >= 0, -1).where(
        values < 0, 1
    )


def _rolling_volatility(frame: pd.DataFrame, window: int) -> pd.Series:
    returns = frame.groupby("ticker", sort=False)["adj_close"].pct_change(
        fill_method=None
    )
    return (
        returns.groupby(frame["ticker"])
        .rolling(window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )


def _rolling_corr(frame: pd.DataFrame, window: int) -> pd.Series:
    return _rolling_pair_corr(
        frame["adj_close"], frame["volume"], frame, window
    )


def _rolling_pair_corr(
    left: pd.Series,
    right: pd.Series,
    frame: pd.DataFrame,
    window: int,
) -> pd.Series:
    parts = []
    for _, ticker_frame in frame.groupby("ticker", sort=False):
        index = ticker_frame.index
        parts.append(
            left.loc[index]
            .rolling(window, min_periods=window)
            .corr(right.loc[index])
        )
    return pd.concat(parts).sort_index()


def _rolling_covariance(
    left: pd.Series,
    right: pd.Series,
    frame: pd.DataFrame,
    window: int,
) -> pd.Series:
    parts = []
    for _, ticker_frame in frame.groupby("ticker", sort=False):
        index = ticker_frame.index
        parts.append(
            left.loc[index]
            .rolling(window, min_periods=window)
            .cov(right.loc[index])
        )
    return pd.concat(parts).sort_index()


def _rolling_mean(values: pd.Series, frame: pd.DataFrame, window: int) -> pd.Series:
    return (
        values.groupby(frame["ticker"])
        .rolling(window, min_periods=window)
        .mean()
        .reset_index(level=0, drop=True)
    )


def _rolling_std(values: pd.Series, frame: pd.DataFrame, window: int) -> pd.Series:
    return (
        values.groupby(frame["ticker"])
        .rolling(window, min_periods=window)
        .std()
        .reset_index(level=0, drop=True)
    )


def _rolling_sum(values: pd.Series, frame: pd.DataFrame, window: int) -> pd.Series:
    return (
        values.groupby(frame["ticker"])
        .rolling(window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
    )


def _sign(values: pd.Series) -> pd.Series:
    return values.gt(0).astype(float) - values.lt(0).astype(float)


def _overnight_gap_rank_product(frame: pd.DataFrame) -> pd.Series:
    previous_high = frame.groupby("ticker", sort=False)["high"].shift(1)
    previous_close = frame.groupby("ticker", sort=False)["close"].shift(1)
    previous_low = frame.groupby("ticker", sort=False)["low"].shift(1)
    return (
        _cross_sectional_rank(frame["open"] - previous_high, frame)
        * _cross_sectional_rank(frame["open"] - previous_close, frame)
        * _cross_sectional_rank(frame["open"] - previous_low, frame)
    )


def _price_position(frame: pd.DataFrame, window: int) -> pd.Series:
    grouped = frame.groupby("ticker", sort=False)
    rolling_high = grouped["high"].transform(
        lambda values: values.rolling(window, min_periods=window).max()
    )
    rolling_low = grouped["low"].transform(
        lambda values: values.rolling(window, min_periods=window).min()
    )
    denominator = (rolling_high - rolling_low).replace(0, pd.NA)
    return (frame["close"] - rolling_low) / denominator


def _rolling_vwap_proxy(frame: pd.DataFrame, window: int) -> pd.Series:
    typical_price = (frame["high"] + frame["low"] + frame["close"]) / 3
    dollar_volume = typical_price * frame["volume"]
    grouped_dollar = dollar_volume.groupby(frame["ticker"])
    grouped_volume = frame["volume"].groupby(frame["ticker"])
    numerator = (
        grouped_dollar.rolling(window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
    )
    denominator = (
        grouped_volume.rolling(window, min_periods=window)
        .sum()
        .reset_index(level=0, drop=True)
        .replace(0, pd.NA)
    )
    return numerator / denominator


def _volume_conditioned_reversal(
    frame: pd.DataFrame, volume_window: int, return_window: int
) -> pd.Series:
    average_volume = (
        frame["volume"]
        .groupby(frame["ticker"])
        .rolling(volume_window, min_periods=volume_window)
        .mean()
        .reset_index(level=0, drop=True)
    )
    reversal = -_grouped_pct_change(frame, return_window)
    active = frame["volume"] > average_volume
    return reversal.where(active, 0.0)
