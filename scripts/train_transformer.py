from __future__ import annotations

import argparse

from equity_transformer.training.config import load_transformer_config
from equity_transformer.training.transformer_trainer import TransformerExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Transformer v1.")
    parser.add_argument("--config", default="configs/transformer.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_transformer_config(parse_args().config)
    _, metrics, history = TransformerExperiment(config).run()
    print(f"Completed {len(history)} epochs")
    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")


if __name__ == "__main__":
    main()
