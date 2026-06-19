from __future__ import annotations

import argparse

from equity_transformer.data.quality import (
    MarketQualityAnalyzer,
    load_market_quality_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze market panel data quality.")
    parser.add_argument("--config", default="configs/data_quality.yaml")
    args = parser.parse_args()
    config = load_market_quality_config(args.config)
    result = MarketQualityAnalyzer(config).run()
    summary = result["summary"]
    print(
        f"Checked {summary['rows']:,} rows for {summary['tickers']} tickers; "
        f"found {summary['issue_count']} quality issues."
    )


if __name__ == "__main__":
    main()
