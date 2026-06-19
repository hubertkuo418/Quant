from __future__ import annotations

import argparse

from equity_transformer.data.csv_import import (
    MarketCsvImporter,
    load_market_csv_import_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import CSV OHLCV files into the canonical market panel."
    )
    parser.add_argument("--config", default="configs/csv_import.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_market_csv_import_config(parse_args().config)
    panel = MarketCsvImporter(config).run()
    print(
        f"Imported {len(panel):,} rows for {panel['ticker'].nunique()} tickers "
        f"to {config.output_path}"
    )


if __name__ == "__main__":
    main()
