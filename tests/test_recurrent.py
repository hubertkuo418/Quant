from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from equity_transformer.datasets.builder import DatasetBuilder
from equity_transformer.datasets.config import DatasetConfig
from equity_transformer.models.recurrent import RecurrentSequenceModel
from equity_transformer.training.config import TrainingConfig
from equity_transformer.training.recurrent_config import (
    RecurrentExperimentConfig,
    RecurrentModelConfig,
)
from equity_transformer.training.recurrent_trainer import RecurrentExperiment


def test_recurrent_model_output_shape() -> None:
    model = RecurrentSequenceModel(
        feature_dim=3,
        output_dim=2,
        model_type="gru",
        hidden_dim=8,
        num_layers=1,
        dropout=0.0,
    )

    assert model(torch.randn(4, 10, 3)).shape == (4, 2)


def test_recurrent_model_rejects_unknown_type() -> None:
    with pytest.raises(ValueError, match="Unsupported"):
        RecurrentSequenceModel(3, 2, "bad", 8, 1, 0.0)


def make_features() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=100)
    parts = []
    for index, ticker in enumerate(("AAA", "BBB", "CCC")):
        time = np.arange(len(dates), dtype=float)
        price = 100 + index * 5 + time * (0.1 + index * 0.03)
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "adj_close": price,
                    "feature_a": np.sin(time / 8),
                    "feature_b": np.cos(time / 10 + index),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def make_dataset_config(tmp_path: Path) -> DatasetConfig:
    dates = pd.bdate_range("2022-01-03", periods=100)
    return DatasetConfig(
        feature_path=tmp_path / "features.parquet",
        output_path=tmp_path / "panel.parquet",
        scaler_path=tmp_path / "metadata" / "scaler.json",
        metadata_dir=tmp_path / "metadata",
        sequence_length=8,
        horizons=(5, 10),
        train_end=dates[49].date().isoformat(),
        validation_end=dates[74].date().isoformat(),
        batch_size=16,
        feature_columns=("feature_a", "feature_b"),
    )


def test_recurrent_experiment_saves_artifacts(tmp_path: Path) -> None:
    dataset_config = make_dataset_config(tmp_path)
    _, datasets = DatasetBuilder(dataset_config).run(make_features())
    config = RecurrentExperimentConfig(
        dataset_config=tmp_path / "unused.yaml",
        artifacts_dir=tmp_path / "recurrent",
        random_seed=11,
        model_type="gru",
        model=RecurrentModelConfig(hidden_dim=8, num_layers=1, dropout=0.0),
        training=TrainingConfig(
            learning_rate=0.005,
            weight_decay=0.0,
            epochs=2,
            patience=2,
            batch_size=16,
            gradient_clip=1.0,
        ),
    )

    predictions, metrics, history = RecurrentExperiment(
        config, dataset_config
    ).run(datasets)

    assert len(history) == 2
    assert not predictions.empty
    assert set(metrics["horizon"]) == {5, 10}
    assert (config.artifacts_dir / "best_model.pt").exists()
