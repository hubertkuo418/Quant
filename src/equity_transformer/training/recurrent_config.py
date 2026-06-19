from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from equity_transformer.training.config import TrainingConfig


@dataclass(frozen=True)
class RecurrentModelConfig:
    hidden_dim: int
    num_layers: int
    dropout: float


@dataclass(frozen=True)
class RecurrentExperimentConfig:
    dataset_config: Path
    artifacts_dir: Path
    random_seed: int
    model_type: str
    model: RecurrentModelConfig
    training: TrainingConfig


def load_recurrent_config(path: str | Path) -> RecurrentExperimentConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    model = payload["model"]
    training = payload["training"]
    return RecurrentExperimentConfig(
        dataset_config=Path(payload["dataset_config"]),
        artifacts_dir=Path(payload["artifacts_dir"]),
        random_seed=int(payload["random_seed"]),
        model_type=str(payload["model_type"]),
        model=RecurrentModelConfig(
            hidden_dim=int(model["hidden_dim"]),
            num_layers=int(model["num_layers"]),
            dropout=float(model["dropout"]),
        ),
        training=TrainingConfig(
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
            batch_size=int(training["batch_size"]),
            gradient_clip=float(training["gradient_clip"]),
        ),
    )
