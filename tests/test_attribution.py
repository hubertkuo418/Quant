from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.backtest.attribution import (
    AttributionConfig,
    AttributionPipeline,
    calculate_return_attribution,
    contribution_concentration_metrics,
    summarize_ticker_attribution,
)


def make_market() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=3)
    rows = []
    for ticker, prices in (("AAA", [100, 110, 121]), ("BBB", [100, 90, 99])):
        for date, price in zip(dates, prices, strict=True):
            rows.append({"date": date, "ticker": ticker, "adj_close": price})
    return pd.DataFrame(rows)


def make_holdings() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=3)[1:]
    return pd.DataFrame(
        [
            {
                "date": date,
                "ticker": ticker,
                "sector": sector,
                "weight": 0.5,
            }
            for date in dates
            for ticker, sector in (("AAA", "Tech"), ("BBB", "Utilities"))
        ]
    )


def test_return_attribution_matches_weight_times_asset_return() -> None:
    attribution = calculate_return_attribution(make_market(), make_holdings())
    first_day = attribution[attribution["date"] == pd.Timestamp("2024-01-02")]

    assert np.isclose(
        first_day.set_index("ticker").loc["AAA", "return_contribution"], 0.05
    )
    assert np.isclose(
        first_day.set_index("ticker").loc["BBB", "return_contribution"], -0.05
    )
    assert set(attribution["sector"]) == {"Tech", "Utilities"}


def test_ticker_summary_reports_contribution_concentration() -> None:
    attribution = calculate_return_attribution(make_market(), make_holdings())
    summary = summarize_ticker_attribution(attribution)
    metrics = contribution_concentration_metrics(summary)

    assert summary["days_held"].eq(2).all()
    assert np.isclose(summary["share_of_absolute_contribution"].sum(), 1.0)
    assert np.isclose(metrics["top_1_absolute_contribution_share"], 0.5)
    assert np.isclose(metrics["top_5_absolute_contribution_share"], 1.0)
    assert np.isclose(metrics["effective_contributors"], 2.0)


def test_attribution_pipeline_writes_artifacts(tmp_path: Path) -> None:
    config = AttributionConfig(
        market_path=tmp_path / "market.parquet",
        holdings_path=tmp_path / "holdings.parquet",
        output_dir=tmp_path / "attribution",
    )

    daily, summary, metrics = AttributionPipeline(config).run(
        make_market(), make_holdings()
    )

    assert not daily.empty
    assert not summary.empty
    assert metrics["effective_contributors"] == 2.0
    assert (config.output_dir / "daily_attribution.parquet").exists()
    assert (config.output_dir / "ticker_summary.csv").exists()
    assert (config.output_dir / "metrics.json").exists()
