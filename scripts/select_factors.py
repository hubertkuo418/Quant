from __future__ import annotations

import argparse

from equity_transformer.factors.selection import (
    FactorSelectionPipeline,
    load_factor_selection_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Select validated factors.")
    parser.add_argument("--config", default="configs/factor_selection.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_factor_selection_config(parse_args().config)
    selected = FactorSelectionPipeline(config).run()
    print(selected[["factor", "selection_score"]].to_string(index=False))


if __name__ == "__main__":
    main()
