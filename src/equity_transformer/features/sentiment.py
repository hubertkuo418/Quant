from __future__ import annotations

import re
from datetime import time
from zoneinfo import ZoneInfo

import pandas as pd

POSITIVE_WORDS = {
    "beat",
    "beats",
    "bullish",
    "gain",
    "gains",
    "growth",
    "improve",
    "improves",
    "outperform",
    "profit",
    "profits",
    "record",
    "strong",
    "surge",
    "upgrade",
}
NEGATIVE_WORDS = {
    "bearish",
    "cut",
    "decline",
    "downgrade",
    "fall",
    "falls",
    "fraud",
    "loss",
    "losses",
    "miss",
    "misses",
    "risk",
    "slump",
    "weak",
    "warning",
}


def score_headline(headline: str) -> float:
    tokens = re.findall(r"[a-z]+", str(headline).lower())
    positive = sum(token in POSITIVE_WORDS for token in tokens)
    negative = sum(token in NEGATIVE_WORDS for token in tokens)
    scored_words = positive + negative
    return (positive - negative) / scored_words if scored_words else 0.0


def aggregate_news(
    news: pd.DataFrame,
    market_dates: pd.Series | pd.Index | None = None,
    market_timezone: str = "America/New_York",
    cutoff_time: str = "09:30",
) -> pd.DataFrame:
    required = {"ticker", "available_at"}
    missing = required.difference(news.columns)
    if missing:
        raise ValueError(f"Missing news columns: {sorted(missing)}")
    if "sentiment_score" not in news.columns and "headline" not in news.columns:
        raise ValueError("News needs either sentiment_score or headline.")

    source = news.copy()
    source["ticker"] = source["ticker"].astype("string").str.strip().str.upper()
    if source["ticker"].isna().any() or source["ticker"].eq("").any():
        raise ValueError("News ticker keys must be valid and non-empty.")
    source["date"] = source["available_at"].map(
        lambda value: _tradable_date(
            value,
            market_timezone=market_timezone,
            cutoff_time=cutoff_time,
        )
    )
    if market_dates is not None:
        source["date"] = _align_to_market_calendar(source["date"], market_dates)
    if "sentiment_score" not in source.columns:
        source["sentiment_score"] = source["headline"].map(score_headline)
    source["sentiment_score"] = pd.to_numeric(
        source["sentiment_score"], errors="coerce"
    )

    return (
        source.dropna(subset=["sentiment_score"])
        .groupby(["date", "ticker"], as_index=False)
        .agg(
            news_sentiment=("sentiment_score", "mean"),
            news_article_count=("sentiment_score", "size"),
        )
    )


def merge_news_features(
    features: pd.DataFrame,
    news: pd.DataFrame,
    market_timezone: str = "America/New_York",
    cutoff_time: str = "09:30",
) -> pd.DataFrame:
    daily_news = aggregate_news(
        news,
        market_dates=features["date"],
        market_timezone=market_timezone,
        cutoff_time=cutoff_time,
    )
    merged = features.merge(daily_news, on=["date", "ticker"], how="left")
    merged["news_sentiment"] = merged["news_sentiment"].fillna(0.0)
    merged["news_article_count"] = merged["news_article_count"].fillna(0).astype(int)
    return merged


def _tradable_date(
    value: object,
    market_timezone: str,
    cutoff_time: str,
) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        return pd.NaT
    zone = ZoneInfo(market_timezone)
    local = (
        timestamp.tz_localize(zone)
        if timestamp.tzinfo is None
        else timestamp.tz_convert(zone)
    )
    cutoff = time.fromisoformat(cutoff_time)
    date = local.tz_localize(None).normalize()
    if local.time().replace(tzinfo=None) > cutoff:
        date += pd.Timedelta(days=1)
    return date


def _align_to_market_calendar(
    dates: pd.Series,
    market_dates: pd.Series | pd.Index,
) -> pd.Series:
    calendar = pd.DatetimeIndex(
        pd.to_datetime(market_dates).dropna().unique()
    ).sort_values()
    positions = calendar.searchsorted(pd.DatetimeIndex(dates), side="left")
    aligned = pd.Series(pd.NaT, index=dates.index, dtype="datetime64[ns]")
    valid = positions < len(calendar)
    aligned.loc[valid] = calendar.take(positions[valid]).to_numpy()
    return aligned
