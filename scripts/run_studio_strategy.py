from __future__ import annotations

import argparse

from equity_transformer.studio.runner import StrategyStudioRunner
from equity_transformer.studio.specs import load_strategy_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Strategy Studio spec.")
    parser.add_argument("--spec", required=True, help="StrategySpec YAML path.")
    parser.add_argument("--runs-root", default="artifacts/studio/runs")
    args = parser.parse_args()
    result = StrategyStudioRunner(args.runs_root).run(load_strategy_spec(args.spec))
    print(f"Completed strategy run {result.run_id}")
    print(f"Artifacts: {result.run_dir}")
    for key, value in result.metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
