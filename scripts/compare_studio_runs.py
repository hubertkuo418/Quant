from __future__ import annotations

import argparse

from equity_transformer.studio.comparison import compare_strategy_runs
from equity_transformer.studio.registry import StrategyRunRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Strategy Studio runs.")
    parser.add_argument("run_ids", nargs="+")
    parser.add_argument("--runs-root", default="artifacts/studio/runs")
    parser.add_argument("--output-dir", default="artifacts/studio/comparisons/latest")
    args = parser.parse_args()
    comparison = compare_strategy_runs(
        args.run_ids,
        StrategyRunRegistry(args.runs_root),
        args.output_dir,
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
