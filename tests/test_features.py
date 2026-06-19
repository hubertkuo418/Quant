from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.features.config import FeatureConfig
from equity_transformer.features.fundamentals import (
    merge_point_in_time_fundamentals,
)
from equity_transformer.features.pipeline import FeaturePipeline
from equity_transformer.features.sentiment import (
    aggregate_news,
    merge_news_features,
    score_headline,
)
from equity_transformer.features.technical import build_technical_features


def make_config(tmp_path: Path) -> FeatureConfig:
    return FeatureConfig(
        market_path=tmp_path / "market.parquet",
        output_path=tmp_path / "features.parquet",
        metadata_dir=tmp_path / "metadata",
        fundamentals_path=None,
        news_path=None,
        return_windows=(1, 5, 20),
        volatility_window=20,
        rsi_window=14,
        moving_average_windows=(20, 60),
        volume_window=20,
        drop_incomplete=False,
    )


def make_market(rows: int = 80) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=rows)
    frames = []
    for ticker, offset in [("AAA", 0.0), ("BBB", 50.0)]:
        close = 100.0 + offset + np.arange(rows, dtype=float)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": close - 0.5,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000 + np.arange(rows) * 10,
                }
            )
        )
    return (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def test_technical_features_are_grouped_and_trailing(tmp_path: Path) -> None:
    features = build_technical_features(make_market(), make_config(tmp_path))
    aaa = features[features["ticker"] == "AAA"].reset_index(drop=True)
    bbb = features[features["ticker"] == "BBB"].reset_index(drop=True)

    assert np.isnan(aaa.loc[0, "return_1d"])
    assert np.isnan(bbb.loc[0, "return_1d"])
    assert np.isclose(aaa.loc[5, "return_5d"], 105 / 100 - 1)
    assert np.isclose(aaa.loc[19, "price_ma_20d_ratio"], 119 / 109.5 - 1)
    assert aaa.loc[14, "rsi_14d"] == 100.0


def test_extended_technical_factor_library(tmp_path: Path) -> None:
    config = replace(
        make_config(tmp_path),
        momentum_windows=(10,),
        volatility_windows=(10,),
        drawdown_windows=(10,),
        atr_window=5,
        bollinger_window=10,
    )
    features = build_technical_features(make_market(), config)
    aaa = features[features["ticker"] == "AAA"].reset_index(drop=True)

    expected_columns = {
        "momentum_10d",
        "volatility_10d",
        "max_drawdown_10d",
        "macd",
        "macd_signal",
        "macd_histogram",
        "atr_5d",
        "atr_5d_ratio",
        "bollinger_percent_b_10d",
        "bollinger_bandwidth_10d",
    }
    assert expected_columns.issubset(features.columns)
    assert np.isclose(aaa.loc[10, "momentum_10d"], 110 / 100 - 1)
    assert aaa.loc[9, "max_drawdown_10d"] == 0.0
    assert aaa.loc[9, "bollinger_bandwidth_10d"] > 0


def test_future_prices_do_not_change_past_features(tmp_path: Path) -> None:
    market = make_market()
    config = make_config(tmp_path)
    original = build_technical_features(market, config)
    cutoff = market["date"].sort_values().unique()[39]

    changed = market.copy()
    changed.loc[changed["date"] > cutoff, "adj_close"] *= 10
    recalculated = build_technical_features(changed, config)
    feature_columns = [column for column in original if column not in market.columns]
    original_past = original.loc[
        original["date"] <= cutoff, feature_columns
    ].reset_index(drop=True)
    recalculated_past = recalculated.loc[
        recalculated["date"] <= cutoff, feature_columns
    ].reset_index(drop=True)

    pd.testing.assert_frame_equal(
        original_past,
        recalculated_past,
    )


def test_fundamentals_are_not_visible_before_available_date(
    tmp_path: Path,
) -> None:
    features = build_technical_features(make_market(10), make_config(tmp_path))
    fundamentals = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "available_at": ["2024-01-08", "2024-01-12"],
            "pe_ratio": [20.0, 22.0],
            "pb_ratio": [3.0, 3.5],
        }
    )

    merged = merge_point_in_time_fundamentals(features, fundamentals)
    aaa = merged[merged["ticker"] == "AAA"].set_index("date")

    assert np.isnan(aaa.loc["2024-01-05", "pe_ratio"])
    assert aaa.loc["2024-01-08", "pe_ratio"] == 20.0
    assert aaa.loc["2024-01-12", "pe_ratio"] == 22.0


def test_news_lexicon_and_daily_aggregation() -> None:
    assert score_headline("Strong growth beats estimates") == 1.0
    assert score_headline("Weak outlook and profit warning") == -1 / 3

    daily = aggregate_news(
        pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "available_at": ["2024-01-08", "2024-01-08"],
                "headline": ["Strong growth", "Profit warning"],
            }
        )
    )

    assert daily.loc[0, "news_article_count"] == 2
    assert np.isclose(daily.loc[0, "news_sentiment"], 0.5)


def test_news_after_open_is_deferred_to_next_market_date(tmp_path: Path) -> None:
    features = build_technical_features(make_market(6), make_config(tmp_path))
    news = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "available_at": [
                "2024-01-05T09:00:00-05:00",
                "2024-01-05T10:00:00-05:00",
            ],
            "sentiment_score": [1.0, -1.0],
        }
    )

    merged = merge_news_features(features, news)
    aaa = merged[merged["ticker"] == "AAA"].set_index("date")

    assert aaa.loc["2024-01-05", "news_sentiment"] == 1.0
    assert aaa.loc["2024-01-08", "news_sentiment"] == -1.0


def test_utc_news_timestamp_uses_market_timezone_cutoff(tmp_path: Path) -> None:
    features = build_technical_features(make_market(6), make_config(tmp_path))
    news = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "available_at": ["2024-01-05T15:00:00Z"],
            "sentiment_score": [0.5],
        }
    )

    merged = merge_news_features(features, news)
    aaa = merged[merged["ticker"] == "AAA"].set_index("date")

    assert aaa.loc["2024-01-05", "news_article_count"] == 0
    assert aaa.loc["2024-01-08", "news_sentiment"] == 0.5


def test_feature_pipeline_writes_panel_and_coverage(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), moving_average_windows=(5, 20))
    news = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "available_at": ["2024-01-08"],
            "headline": ["Strong profit growth"],
        }
    )

    features = FeaturePipeline(config).run(market=make_market(30), news=news)

    assert config.output_path.exists()
    assert len(list(config.metadata_dir.glob("features_*.json"))) == 1
    assert {"return_5d", "pe_ratio", "news_sentiment"}.issubset(features.columns)
    assert features.loc[
        (features["ticker"] == "AAA")
        & (features["date"] == pd.Timestamp("2024-01-08")),
        "news_article_count",
    ].item() == 1
