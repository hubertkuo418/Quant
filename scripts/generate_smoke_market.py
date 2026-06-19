from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_TICKERS = ("AAPL", "MSFT", "NVDA", "AMZN", "META", "JPM", "XOM", "SPY")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic synthetic OHLCV for pipeline smoke tests."
    )
    parser.add_argument("--output", default="data/processed/market_panel.parquet")
    parser.add_argument(
        "--manifest", default="data/metadata/smoke_market_manifest.json"
    )
    parser.add_argument("--start", default="2023-01-02")
    parser.add_argument("--days", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_TICKERS))
    return parser.parse_args()


def generate_smoke_market(
    start: str,
    days: int,
    tickers: list[str],
    seed: int,
) -> pd.DataFrame:
    if days < 2:
        raise ValueError("days must be at least 2.")
    if len(set(tickers)) != len(tickers):
        raise ValueError("tickers must be unique.")
    dates = pd.bdate_range(start, periods=days)
    rng = np.random.default_rng(seed)
    parts = []
    market_cycle = np.sin(np.arange(days) / 35) * 0.004
    for index, ticker in enumerate(tickers):
        idiosyncratic = rng.normal(0, 0.008 + index * 0.0005, days)
        drift = 0.00015 + index * 0.00003
        daily_return = np.clip(drift + market_cycle + idiosyncratic, -0.15, 0.15)
        close = (80 + index * 8) * np.cumprod(1 + daily_return)
        prior_close = np.concatenate(([close[0]], close[:-1]))
        open_price = prior_close * (1 + rng.normal(0, 0.002, days))
        spread = np.abs(rng.normal(0.004, 0.001, days))
        high = np.maximum(open_price, close) * (1 + spread)
        low = np.minimum(open_price, close) * (1 - spread)
        volume = rng.integers(1_000_000, 8_000_000, days) * (index + 1)
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "adj_close": close,
                    "volume": volume,
                }
            )
        )
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"])


def main() -> None:
    args = parse_args()
    panel = generate_smoke_market(args.start, args.days, args.tickers, args.seed)
    output = Path(args.output)
    manifest = Path(args.manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(output, index=False)
    manifest.write_text(
        json.dumps(
            {
                "synthetic": True,
                "purpose": "pipeline smoke testing only",
                "seed": args.seed,
                "start": args.start,
                "days": args.days,
                "tickers": args.tickers,
                "rows": len(panel),
                "output": str(output),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(panel):,} synthetic rows to {output}")


if __name__ == "__main__":
    main()
