from __future__ import annotations

import argparse

from equity_transformer.datasets.builder import DatasetBuilder
from equity_transformer.datasets.config import load_dataset_config
from equity_transformer.datasets.sequence import make_dataloader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build model-ready sequences.")
    parser.add_argument("--config", default="configs/dataset.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_dataset_config(parse_args().config)
    _, datasets = DatasetBuilder(config).run()
    loaders = {
        split: make_dataloader(
            dataset,
            batch_size=config.batch_size,
            shuffle=split == "train",
        )
        for split, dataset in datasets.items()
    }
    counts = ", ".join(f"{split}={len(data)}" for split, data in datasets.items())
    print(f"Built datasets: {counts}")
    for split, loader in loaders.items():
        if len(loader):
            features, targets = next(iter(loader))
            print(f"{split}: X={tuple(features.shape)}, y={tuple(targets.shape)}")


if __name__ == "__main__":
    main()
