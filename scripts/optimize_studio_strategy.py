from __future__ import annotations

import argparse

from equity_transformer.studio.optimizer import (
    StrategyOptimizer,
    load_optimization_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize a Strategy Studio spec.")
    parser.add_argument("--config", default="configs/studio_optimizer.yaml")
    args = parser.parse_args()
    results = StrategyOptimizer(load_optimization_config(args.config)).run()
    print(results.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
