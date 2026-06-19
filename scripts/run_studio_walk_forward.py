from __future__ import annotations

import argparse

from equity_transformer.studio.walk_forward import (
    WalkForwardEvaluator,
    load_walk_forward_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run rolling OOS strategy evaluation")
    parser.add_argument(
        "--config",
        default="configs/studio_walk_forward.yaml",
        help="Path to walk-forward YAML config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = WalkForwardEvaluator(load_walk_forward_config(args.config)).run()
    print(f"Wrote {len(result.folds)} folds to {result.output_dir}")
    print(f"OOS Sharpe: {result.metrics['sharpe_ratio']:.4f}")


if __name__ == "__main__":
    main()
