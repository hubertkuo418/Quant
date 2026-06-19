from __future__ import annotations

import argparse
import logging

from equity_transformer.data.config import load_data_config
from equity_transformer.data.pipeline import MarketDataPipeline
from equity_transformer.data.providers import create_provider


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and clean daily OHLCV data.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD).")
    parser.add_argument("--end", help="Override exclusive end date (YYYY-MM-DD).")
    parser.add_argument("--tickers", nargs="+", help="Override configured universe.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = load_data_config(args.config)
    provider = create_provider(config.provider)
    pipeline = MarketDataPipeline(config, provider)
    tickers = tuple(ticker.upper() for ticker in args.tickers) if args.tickers else None
    panel = pipeline.run(tickers=tickers, start=args.start, end=args.end)
    print(
        f"Saved {len(panel):,} rows for {panel['ticker'].nunique()} tickers "
        f"to {config.processed_path}"
    )


if __name__ == "__main__":
    main()
