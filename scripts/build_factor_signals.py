from __future__ import annotations

import argparse

from equity_transformer.factors.signals import (
    FactorSignalPipeline,
    load_factor_signal_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build IC-weighted factor signals.")
    parser.add_argument("--config", default="configs/factor_signals.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_factor_signal_config(parse_args().config)
    signals = FactorSignalPipeline(config).run()
    print(
        f"Saved {len(signals):,} factor signal rows across "
        f"{signals[config.date_column].nunique() if not signals.empty else 0} dates."
    )


if __name__ == "__main__":
    main()
