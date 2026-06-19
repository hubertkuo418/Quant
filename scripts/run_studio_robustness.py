from __future__ import annotations

import argparse

from equity_transformer.studio.robustness import (
    RobustnessEvaluator,
    load_robustness_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run unified OOS robustness checks")
    parser.add_argument("--config", default="configs/studio_robustness.yaml")
    args = parser.parse_args()
    result = RobustnessEvaluator(load_robustness_config(args.config)).run()
    print(f"Wrote {len(result.scenarios)} scenarios to {result.output_dir}")
    print(f"Worst Sharpe: {result.aggregate['worst_sharpe']:.4f}")
    print(f"Pass rate: {result.aggregate['pass_rate']:.2%}")


if __name__ == "__main__":
    main()
