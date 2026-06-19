from __future__ import annotations

import argparse

from equity_transformer.alphas.config import load_alpha_config
from equity_transformer.alphas.pipeline import AlphaPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build formulaic alpha panel.")
    parser.add_argument("--config", default="configs/alphas.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_alpha_config(parse_args().config)
    panel = AlphaPipeline(config).run()
    print(f"Saved {len(panel):,} alpha rows to {config.output_path}")


if __name__ == "__main__":
    main()
