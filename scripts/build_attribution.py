from __future__ import annotations

import argparse

from equity_transformer.backtest.attribution import (
    AttributionPipeline,
    load_attribution_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build portfolio return attribution.")
    parser.add_argument("--config", default="configs/attribution.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_attribution_config(parse_args().config)
    daily, summary, metrics = AttributionPipeline(config).run()
    print(f"Wrote {len(daily)} daily rows for {len(summary)} tickers.")
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
