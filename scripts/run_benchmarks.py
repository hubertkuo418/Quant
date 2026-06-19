from __future__ import annotations

import argparse

from equity_transformer.backtest.benchmark import (
    BenchmarkPipeline,
    load_benchmark_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run benchmark backtests.")
    parser.add_argument("--config", default="configs/benchmarks.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_benchmark_config(parse_args().config)
    comparison = BenchmarkPipeline(config).run()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
