from __future__ import annotations

import argparse

from equity_transformer.backtest.config import load_backtest_config
from equity_transformer.backtest.engine import BacktestEngine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a portfolio backtest.")
    parser.add_argument("--config", default="configs/backtest.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_backtest_config(parse_args().config)
    _, _, metrics = BacktestEngine(config).run()
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
