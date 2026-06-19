from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_transformer.backtest.config import BacktestConfig
from equity_transformer.backtest.engine import BacktestEngine


@dataclass(frozen=True)
class BenchmarkConfig:
    market_path: Path
    output_dir: Path
    rebalance_frequency: str
    buy_and_hold_ticker: str | None
    momentum_window: int
    momentum_top_k: int
    initial_capital: float
    commission_bps: float
    slippage_bps: float
    annualization_factor: int
    risk_free_rate: float


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return BenchmarkConfig(
        market_path=Path(payload["market_path"]),
        output_dir=Path(payload["output_dir"]),
        rebalance_frequency=str(payload["rebalance_frequency"]),
        buy_and_hold_ticker=payload.get("buy_and_hold_ticker"),
        momentum_window=int(payload["momentum_window"]),
        momentum_top_k=int(payload["momentum_top_k"]),
        initial_capital=float(payload["initial_capital"]),
        commission_bps=float(payload["commission_bps"]),
        slippage_bps=float(payload["slippage_bps"]),
        annualization_factor=int(payload["annualization_factor"]),
        risk_free_rate=float(payload["risk_free_rate"]),
    )


def equal_weight_universe_weights(
    market: pd.DataFrame, rebalance_frequency: str
) -> pd.DataFrame:
    frame = market[["date", "ticker"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rebalance_dates = _rebalance_dates(frame["date"], rebalance_frequency)
    frame = frame[frame["date"].isin(rebalance_dates)]
    parts = []
    for date, daily in frame.groupby("date"):
        tickers = sorted(daily["ticker"].unique())
        if not tickers:
            continue
        weight = 1 / len(tickers)
        parts.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": tickers,
                    "weight": weight,
                    "side": "long",
                    "score": 1.0,
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["date", "ticker", "weight", "side", "score"])
    return pd.concat(parts, ignore_index=True)


def buy_and_hold_weights(market: pd.DataFrame, ticker: str) -> pd.DataFrame:
    frame = market[market["ticker"] == ticker].copy()
    if frame.empty:
        return pd.DataFrame(columns=["date", "ticker", "weight", "side", "score"])
    first_date = pd.to_datetime(frame["date"]).min()
    return pd.DataFrame(
        {
            "date": [first_date],
            "ticker": [ticker],
            "weight": [1.0],
            "side": ["long"],
            "score": [1.0],
        }
    )


def momentum_top_k_weights(
    market: pd.DataFrame,
    window: int,
    top_k: int,
    rebalance_frequency: str,
) -> pd.DataFrame:
    frame = market[["date", "ticker", "adj_close"]].copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["ticker", "date"])
    frame["score"] = frame.groupby("ticker")["adj_close"].pct_change(
        window, fill_method=None
    )
    rebalance_dates = _rebalance_dates(frame["date"], rebalance_frequency)
    frame = frame[frame["date"].isin(rebalance_dates)].dropna(subset=["score"])
    parts = []
    for date, daily in frame.groupby("date"):
        selected = daily.nlargest(top_k, "score")
        if selected.empty:
            continue
        parts.append(
            pd.DataFrame(
                {
                    "date": date,
                    "ticker": selected["ticker"].to_list(),
                    "weight": 1 / len(selected),
                    "side": "long",
                    "score": selected["score"].to_list(),
                }
            )
        )
    if not parts:
        return pd.DataFrame(columns=["date", "ticker", "weight", "side", "score"])
    return pd.concat(parts, ignore_index=True)


class BenchmarkPipeline:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self, market: pd.DataFrame | None = None) -> pd.DataFrame:
        market_frame = (
            market.copy()
            if market is not None
            else pd.read_parquet(self.config.market_path)
        )
        benchmarks = {
            "equal_weight_universe": equal_weight_universe_weights(
                market_frame, self.config.rebalance_frequency
            )
        }
        if self.config.buy_and_hold_ticker:
            benchmark = buy_and_hold_weights(
                market_frame, self.config.buy_and_hold_ticker
            )
            if not benchmark.empty:
                name = f"buy_and_hold_{self.config.buy_and_hold_ticker}"
                benchmarks[name] = benchmark
        momentum = momentum_top_k_weights(
            market_frame,
            self.config.momentum_window,
            self.config.momentum_top_k,
            self.config.rebalance_frequency,
        )
        if not momentum.empty:
            benchmarks[
                f"momentum_{self.config.momentum_window}d_top_{self.config.momentum_top_k}"
            ] = momentum

        rows = []
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for name, weights in benchmarks.items():
            engine_config = BacktestConfig(
                market_path=self.config.market_path,
                weights_path=self.config.output_dir / f"{name}_weights.parquet",
                output_dir=self.config.output_dir / name,
                initial_capital=self.config.initial_capital,
                commission_bps=self.config.commission_bps,
                slippage_bps=self.config.slippage_bps,
                annualization_factor=self.config.annualization_factor,
                risk_free_rate=self.config.risk_free_rate,
            )
            weights.to_parquet(engine_config.weights_path, index=False)
            _, _, metrics = BacktestEngine(engine_config).run(market_frame, weights)
            rows.append({"benchmark": name, **metrics})

        comparison = pd.DataFrame(rows).sort_values(
            "sharpe_ratio", ascending=False, na_position="last"
        )
        comparison.to_csv(self.config.output_dir / "comparison.csv", index=False)
        (self.config.output_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "benchmarks": sorted(benchmarks),
                    "rebalance_frequency": self.config.rebalance_frequency,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return comparison


def _rebalance_dates(dates: pd.Series, frequency: str) -> set[pd.Timestamp]:
    unique_dates = pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values()
    if unique_dates.empty:
        return set()
    if frequency.lower() in {"daily", "d"}:
        return set(unique_dates)
    grouped = unique_dates.groupby(unique_dates.dt.to_period(frequency))
    return set(grouped.max())
