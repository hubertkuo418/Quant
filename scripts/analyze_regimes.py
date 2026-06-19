from __future__ import annotations

import argparse

from equity_transformer.backtest.regime import (
    RegimeAnalysisPipeline,
    load_regime_analysis_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze strategy market regimes.")
    parser.add_argument("--config", default="configs/regime.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_regime_analysis_config(parse_args().config)
    regimes, performance = RegimeAnalysisPipeline(config).run()
    print(f"Classified {len(regimes)} dates across {len(performance)} regimes.")


if __name__ == "__main__":
    main()
