from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class AttributionConfig:
    market_path: Path
    holdings_path: Path
    output_dir: Path


def load_attribution_config(path: str | Path) -> AttributionConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return AttributionConfig(
        market_path=Path(payload["market_path"]),
        holdings_path=Path(payload["holdings_path"]),
        output_dir=Path(payload["output_dir"]),
    )


def calculate_return_attribution(
    market: pd.DataFrame,
    holdings: pd.DataFrame,
) -> pd.DataFrame:
    market_required = {"date", "ticker", "adj_close"}
    holdings_required = {"date", "ticker", "weight"}
    missing_market = market_required.difference(market.columns)
    missing_holdings = holdings_required.difference(holdings.columns)
    if missing_market:
        raise ValueError(f"Market panel missing columns: {sorted(missing_market)}")
    if missing_holdings:
        raise ValueError(f"Holdings missing columns: {sorted(missing_holdings)}")

    prices = market[["date", "ticker", "adj_close"]].copy()
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.sort_values(["ticker", "date"])
    prices["asset_return"] = prices.groupby("ticker")["adj_close"].pct_change()
    position_columns = ["date", "ticker", "weight"]
    if "sector" in holdings.columns:
        position_columns.append("sector")
    positions = holdings[position_columns].copy()
    positions["date"] = pd.to_datetime(positions["date"])
    attribution = positions.merge(
        prices[["date", "ticker", "asset_return"]],
        on=["date", "ticker"],
        how="left",
        validate="many_to_one",
    )
    attribution["asset_return"] = attribution["asset_return"].fillna(0.0)
    attribution["return_contribution"] = (
        attribution["weight"] * attribution["asset_return"]
    )
    attribution["absolute_contribution"] = attribution["return_contribution"].abs()
    return attribution.sort_values(["date", "ticker"]).reset_index(drop=True)


def summarize_ticker_attribution(attribution: pd.DataFrame) -> pd.DataFrame:
    required = {
        "date",
        "ticker",
        "weight",
        "return_contribution",
        "absolute_contribution",
    }
    missing = required.difference(attribution.columns)
    if missing:
        raise ValueError(f"Attribution missing columns: {sorted(missing)}")
    if attribution.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "total_contribution",
                "absolute_contribution",
                "average_weight",
                "days_held",
                "share_of_absolute_contribution",
                "contribution_rank",
            ]
        )

    summary = (
        attribution.groupby("ticker", as_index=False)
        .agg(
            total_contribution=("return_contribution", "sum"),
            absolute_contribution=("absolute_contribution", "sum"),
            average_weight=("weight", "mean"),
            days_held=("date", "nunique"),
        )
        .sort_values("total_contribution", ascending=False)
        .reset_index(drop=True)
    )
    total_absolute = float(summary["absolute_contribution"].sum())
    summary["share_of_absolute_contribution"] = (
        summary["absolute_contribution"] / total_absolute
        if total_absolute > 0
        else 0.0
    )
    summary["contribution_rank"] = range(1, len(summary) + 1)
    return summary


def contribution_concentration_metrics(summary: pd.DataFrame) -> dict[str, float]:
    if summary.empty:
        return {
            "top_1_absolute_contribution_share": 0.0,
            "top_5_absolute_contribution_share": 0.0,
            "effective_contributors": 0.0,
        }
    shares = summary["share_of_absolute_contribution"].sort_values(ascending=False)
    squared_sum = float(shares.pow(2).sum())
    return {
        "top_1_absolute_contribution_share": float(shares.head(1).sum()),
        "top_5_absolute_contribution_share": float(shares.head(5).sum()),
        "effective_contributors": 1 / squared_sum if squared_sum > 0 else 0.0,
    }


class AttributionPipeline:
    def __init__(self, config: AttributionConfig) -> None:
        self.config = config

    def run(
        self,
        market: pd.DataFrame | None = None,
        holdings: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        market_frame = (
            market.copy()
            if market is not None
            else pd.read_parquet(self.config.market_path)
        )
        holding_frame = (
            holdings.copy()
            if holdings is not None
            else pd.read_parquet(self.config.holdings_path)
        )
        daily = calculate_return_attribution(market_frame, holding_frame)
        summary = summarize_ticker_attribution(daily)
        metrics = contribution_concentration_metrics(summary)
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        daily.to_parquet(
            self.config.output_dir / "daily_attribution.parquet", index=False
        )
        summary.to_csv(self.config.output_dir / "ticker_summary.csv", index=False)
        (self.config.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        return daily, summary, metrics
