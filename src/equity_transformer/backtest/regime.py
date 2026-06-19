from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class RegimeAnalysisConfig:
    market_path: Path
    equity_path: Path
    output_dir: Path
    benchmark_ticker: str
    trend_window: int
    volatility_window: int
    threshold_window: int
    annualization_factor: int


def load_regime_analysis_config(path: str | Path) -> RegimeAnalysisConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return RegimeAnalysisConfig(
        market_path=Path(payload["market_path"]),
        equity_path=Path(payload["equity_path"]),
        output_dir=Path(payload["output_dir"]),
        benchmark_ticker=str(payload["benchmark_ticker"]),
        trend_window=int(payload.get("trend_window", 60)),
        volatility_window=int(payload.get("volatility_window", 20)),
        threshold_window=int(payload.get("threshold_window", 252)),
        annualization_factor=int(payload.get("annualization_factor", 252)),
    )


def classify_market_regimes(
    market: pd.DataFrame,
    benchmark_ticker: str,
    trend_window: int,
    volatility_window: int,
    threshold_window: int,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    required = {"date", "ticker", "adj_close"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(f"Market panel missing columns: {sorted(missing)}")
    if min(trend_window, volatility_window, threshold_window) <= 0:
        raise ValueError("Regime windows must be positive.")

    frame = market.loc[
        market["ticker"] == benchmark_ticker,
        ["date", "adj_close"],
    ].copy()
    if frame.empty:
        raise ValueError(f"Benchmark ticker not found: {benchmark_ticker}")
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values("date").drop_duplicates("date", keep="last")
    frame["benchmark_return"] = frame["adj_close"].pct_change()
    frame["trend_return"] = frame["adj_close"].pct_change(trend_window)
    frame["annual_volatility"] = (
        frame["benchmark_return"]
        .rolling(volatility_window, min_periods=volatility_window)
        .std(ddof=1)
        * np.sqrt(annualization_factor)
    )
    frame["volatility_threshold"] = (
        frame["annual_volatility"]
        .shift(1)
        .rolling(threshold_window, min_periods=volatility_window)
        .median()
    )
    complete = frame.dropna(
        subset=["trend_return", "annual_volatility", "volatility_threshold"]
    ).copy()
    trend = np.where(complete["trend_return"] >= 0, "bull", "bear")
    volatility = np.where(
        complete["annual_volatility"] >= complete["volatility_threshold"],
        "high_vol",
        "low_vol",
    )
    complete["trend_regime"] = trend
    complete["volatility_regime"] = volatility
    complete["regime"] = complete["trend_regime"] + "_" + complete["volatility_regime"]
    return complete.reset_index(drop=True)


def summarize_regime_performance(
    equity_curve: pd.DataFrame,
    regimes: pd.DataFrame,
    annualization_factor: int = 252,
) -> pd.DataFrame:
    required_equity = {"date", "net_return"}
    missing = required_equity.difference(equity_curve.columns)
    if missing:
        raise ValueError(f"Equity curve missing columns: {sorted(missing)}")
    if "regime" not in regimes.columns:
        raise ValueError("Regime frame missing column: regime")

    equity = equity_curve[["date", "net_return"]].copy()
    equity["date"] = pd.to_datetime(equity["date"])
    regime_frame = regimes[["date", "regime"]].copy()
    regime_frame["date"] = pd.to_datetime(regime_frame["date"])
    merged = equity.merge(regime_frame, on="date", how="inner").dropna(
        subset=["net_return"]
    )
    rows = []
    for regime, group in merged.groupby("regime", sort=True):
        returns = group["net_return"].astype(float)
        observations = len(returns)
        total_return = float((1 + returns).prod() - 1)
        annual_return = float(
            (1 + total_return) ** (annualization_factor / observations) - 1
        )
        daily_volatility = returns.std(ddof=1)
        annual_volatility = float(daily_volatility * np.sqrt(annualization_factor))
        sharpe = (
            float(returns.mean() / daily_volatility * np.sqrt(annualization_factor))
            if daily_volatility > 0
            else float("nan")
        )
        regime_nav = (1 + returns).cumprod()
        max_drawdown = float((regime_nav / regime_nav.cummax() - 1).min())
        rows.append(
            {
                "regime": regime,
                "observations": observations,
                "start_date": group["date"].min(),
                "end_date": group["date"].max(),
                "total_return": total_return,
                "annual_return": annual_return,
                "annual_volatility": annual_volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown,
                "win_rate": float((returns > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


class RegimeAnalysisPipeline:
    def __init__(self, config: RegimeAnalysisConfig) -> None:
        self.config = config

    def run(
        self,
        market: pd.DataFrame | None = None,
        equity_curve: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        market_frame = (
            market.copy()
            if market is not None
            else pd.read_parquet(self.config.market_path)
        )
        equity = (
            equity_curve.copy()
            if equity_curve is not None
            else pd.read_csv(self.config.equity_path)
        )
        regimes = classify_market_regimes(
            market_frame,
            self.config.benchmark_ticker,
            self.config.trend_window,
            self.config.volatility_window,
            self.config.threshold_window,
            self.config.annualization_factor,
        )
        performance = summarize_regime_performance(
            equity,
            regimes,
            self.config.annualization_factor,
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        regimes.to_csv(self.config.output_dir / "daily_regimes.csv", index=False)
        performance.to_csv(
            self.config.output_dir / "regime_performance.csv", index=False
        )
        (self.config.output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "benchmark_ticker": self.config.benchmark_ticker,
                    "trend_window": self.config.trend_window,
                    "volatility_window": self.config.volatility_window,
                    "threshold_window": self.config.threshold_window,
                    "classified_dates": len(regimes),
                    "regimes": sorted(regimes["regime"].unique().tolist()),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return regimes, performance
