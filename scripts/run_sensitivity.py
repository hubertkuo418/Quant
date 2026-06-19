from __future__ import annotations

import argparse

from equity_transformer.backtest.sensitivity import (
    SensitivityPipeline,
    load_sensitivity_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backtest sensitivity scenarios.")
    parser.add_argument("--config", default="configs/sensitivity.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_sensitivity_config(parse_args().config)
    comparison = SensitivityPipeline(config).run()
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
