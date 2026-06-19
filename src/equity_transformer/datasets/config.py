from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    feature_path: Path
    output_path: Path
    scaler_path: Path
    metadata_dir: Path
    sequence_length: int
    horizons: tuple[int, ...]
    train_end: str
    validation_end: str
    batch_size: int
    feature_columns: tuple[str, ...]

    @property
    def target_columns(self) -> tuple[str, ...]:
        return tuple(f"target_{horizon}d" for horizon in self.horizons)


def load_dataset_config(path: str | Path) -> DatasetConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)

    return DatasetConfig(
        feature_path=Path(payload["feature_path"]),
        output_path=Path(payload["output_path"]),
        scaler_path=Path(payload["scaler_path"]),
        metadata_dir=Path(payload["metadata_dir"]),
        sequence_length=int(payload["sequence_length"]),
        horizons=tuple(int(value) for value in payload["horizons"]),
        train_end=str(payload["train_end"]),
        validation_end=str(payload["validation_end"]),
        batch_size=int(payload["batch_size"]),
        feature_columns=tuple(payload["feature_columns"]),
    )
