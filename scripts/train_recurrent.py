from __future__ import annotations

import argparse

from equity_transformer.training.recurrent_config import load_recurrent_config
from equity_transformer.training.recurrent_trainer import RecurrentExperiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RNN/LSTM/GRU baseline.")
    parser.add_argument("--config", default="configs/recurrent.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_recurrent_config(parse_args().config)
    _, metrics, history = RecurrentExperiment(config).run()
    print(f"Completed {len(history)} epochs")
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
