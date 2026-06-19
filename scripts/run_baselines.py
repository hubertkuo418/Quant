from __future__ import annotations

import argparse

from equity_transformer.baselines.config import load_baseline_config
from equity_transformer.baselines.runner import BaselineExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate baselines.")
    parser.add_argument("--config", default="configs/baselines.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_baseline_config(parse_args().config)
    _, metrics = BaselineExperiment(config).run()
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
