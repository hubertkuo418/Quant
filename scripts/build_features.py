from __future__ import annotations

import argparse

from equity_transformer.features.config import load_feature_config
from equity_transformer.features.pipeline import FeaturePipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build point-in-time equity features.")
    parser.add_argument("--config", default="configs/features.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_feature_config(parse_args().config)
    features = FeaturePipeline(config).run()
    print(
        f"Saved {len(features):,} rows with {len(features.columns):,} columns "
        f"to {config.output_path}"
    )


if __name__ == "__main__":
    main()
