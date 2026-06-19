from __future__ import annotations

import argparse

from equity_transformer.reporting.equity_comparison import (
    EquityComparisonPipeline,
    load_equity_comparison_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare portfolio equity curves over their common period."
    )
    parser.add_argument("--config", default="configs/equity_comparison.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_equity_comparison_config(parse_args().config)
    comparison = EquityComparisonPipeline(config).run()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
