from __future__ import annotations

import argparse

from equity_transformer.factors.panel import (
    FactorPanelPipeline,
    load_factor_panel_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge feature and alpha panels.")
    parser.add_argument("--config", default="configs/factor_panel.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_factor_panel_config(parse_args().config)
    panel = FactorPanelPipeline(config).run()
    print(f"Saved {len(panel):,} factor rows to {config.output_path}")


if __name__ == "__main__":
    main()
