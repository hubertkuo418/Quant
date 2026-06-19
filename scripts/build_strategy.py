from __future__ import annotations

import argparse

from equity_transformer.strategies.config import load_strategy_config
from equity_transformer.strategies.pipeline import StrategyPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build strategy target weights.")
    parser.add_argument("--config", default="configs/strategy.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_strategy_config(parse_args().config)
    weights = StrategyPipeline(config).run()
    print(
        f"Saved {len(weights):,} target weights across "
        f"{weights['date'].nunique() if not weights.empty else 0} rebalance dates."
    )


if __name__ == "__main__":
    main()
