from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class TransformerModelConfig:
    d_model: int
    num_heads: int
    num_layers: int
    feedforward_dim: int
    dropout: float


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    batch_size: int
    gradient_clip: float
    correlation_loss_weight: float = 0.0


@dataclass(frozen=True)
class TransformerExperimentConfig:
    dataset_config: Path
    artifacts_dir: Path
    random_seed: int
    target_horizon: int | None
    model: TransformerModelConfig
    training: TrainingConfig


def load_transformer_config(path: str | Path) -> TransformerExperimentConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    model = payload["model"]
    training = payload["training"]
    return TransformerExperimentConfig(
        dataset_config=Path(payload["dataset_config"]),
        artifacts_dir=Path(payload["artifacts_dir"]),
        random_seed=int(payload["random_seed"]),
        target_horizon=(
            int(payload["target_horizon"])
            if payload.get("target_horizon") is not None
            else None
        ),
        model=TransformerModelConfig(
            d_model=int(model["d_model"]),
            num_heads=int(model["num_heads"]),
            num_layers=int(model["num_layers"]),
            feedforward_dim=int(model["feedforward_dim"]),
            dropout=float(model["dropout"]),
        ),
        training=TrainingConfig(
            learning_rate=float(training["learning_rate"]),
            weight_decay=float(training["weight_decay"]),
            epochs=int(training["epochs"]),
            patience=int(training["patience"]),
            batch_size=int(training["batch_size"]),
            gradient_clip=float(training["gradient_clip"]),
            correlation_loss_weight=float(
                training.get("correlation_loss_weight", 0.0)
            ),
        ),
    )
