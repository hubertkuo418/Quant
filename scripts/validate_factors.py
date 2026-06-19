from __future__ import annotations

import argparse

from equity_transformer.factors.config import load_factor_validation_config
from equity_transformer.factors.validation import FactorValidationPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate factor predictive power.")
    parser.add_argument("--config", default="configs/factors.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_factor_validation_config(parse_args().config)
    outputs = FactorValidationPipeline(config).run()
    summary = outputs["ic_summary"]
    print(summary.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
