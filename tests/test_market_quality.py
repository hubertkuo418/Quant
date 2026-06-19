from __future__ import annotations

from pathlib import Path

import pandas as pd

from equity_transformer.data.quality import MarketQualityAnalyzer, MarketQualityConfig


def make_market() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    rows = []
    for ticker, selected, closes, volumes in (
        ("AAA", dates, [100.0, 100.0, 100.0, 130.0], [100, 100, 0, 100]),
        ("BBB", dates[:2], [50.0, 51.0], [100, 100]),
    ):
        for date, close, volume in zip(selected, closes, volumes, strict=True):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                }
            )
    return pd.DataFrame(rows)


def test_market_quality_reports_coverage_staleness_and_volume(tmp_path: Path) -> None:
    config = MarketQualityConfig(
        market_path=tmp_path / "market.parquet",
        output_dir=tmp_path / "quality",
        min_calendar_coverage=0.75,
        max_zero_volume_rate=0.10,
        max_stale_price_run=1,
        max_abs_daily_return=0.25,
    )

    result = MarketQualityAnalyzer(config).run(make_market())
    metrics = result["per_ticker"].set_index("ticker")
    issues = result["issues"]

    assert metrics.loc["BBB", "calendar_coverage"] == 0.5
    assert metrics.loc["AAA", "longest_stale_price_run"] == 2
    assert set(issues["metric"]) == {
        "calendar_coverage",
        "zero_volume_rate",
        "longest_stale_price_run",
        "max_abs_daily_return",
    }
    assert (config.output_dir / "per_ticker.csv").exists()
    assert (config.output_dir / "issues.csv").exists()
    assert (config.output_dir / "summary.json").exists()


def test_market_quality_can_return_empty_issue_table(tmp_path: Path) -> None:
    config = MarketQualityConfig(
        market_path=tmp_path / "market.parquet",
        output_dir=tmp_path / "quality",
        min_calendar_coverage=0,
        max_zero_volume_rate=1,
        max_stale_price_run=10,
        max_abs_daily_return=1,
    )

    issues = MarketQualityAnalyzer(config).run(make_market())["issues"]

    assert issues.empty
    assert issues.columns.tolist() == [
        "ticker",
        "metric",
        "value",
        "condition",
        "threshold",
    ]
