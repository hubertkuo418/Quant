from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from equity_transformer.datasets.builder import DatasetBuilder
from equity_transformer.datasets.config import DatasetConfig
from equity_transformer.models.transformer import TimeSeriesTransformer
from equity_transformer.training.config import (
    TrainingConfig,
    TransformerExperimentConfig,
    TransformerModelConfig,
)
from equity_transformer.training.transformer_trainer import (
    CorrelationAwareLoss,
    DateGroupedBatchSampler,
    TransformerExperiment,
)


def make_model() -> TimeSeriesTransformer:
    return TimeSeriesTransformer(
        feature_dim=3,
        sequence_length=8,
        output_dim=1,
        d_model=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )


def test_transformer_output_shape() -> None:
    prediction = make_model()(torch.randn(5, 8, 3))
    assert prediction.shape == (5, 1)


def test_transformer_can_overfit_a_tiny_batch() -> None:
    torch.manual_seed(3)
    model = make_model()
    features = torch.randn(12, 8, 3)
    targets = features[:, -1, 0] * 0.5 - features[:, -1, 1] * 0.2
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    loss_function = nn.MSELoss()

    with torch.no_grad():
        initial_loss = loss_function(model(features).squeeze(-1), targets).item()
    for _ in range(80):
        optimizer.zero_grad()
        loss = loss_function(model(features).squeeze(-1), targets)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = loss_function(model(features).squeeze(-1), targets).item()

    assert final_loss < initial_loss * 0.1


def test_correlation_aware_loss_rewards_aligned_cross_section() -> None:
    target = torch.tensor([[-1.0], [0.0], [1.0]])
    aligned = target.clone().requires_grad_(True)
    reversed_prediction = (-target).requires_grad_(True)
    loss_function = CorrelationAwareLoss(correlation_weight=0.5)

    aligned_loss = loss_function(aligned, target)
    reversed_loss = loss_function(reversed_prediction, target)
    aligned_loss.backward()

    assert aligned_loss < reversed_loss
    assert aligned.grad is not None


def test_correlation_aware_loss_handles_constant_targets() -> None:
    prediction = torch.tensor([[0.1], [0.2], [0.3]], requires_grad=True)
    target = torch.ones((3, 1))

    loss = CorrelationAwareLoss(0.5)(prediction, target)
    loss.backward()

    assert torch.isfinite(loss)
    assert prediction.grad is not None


def make_dataset_config(tmp_path: Path) -> DatasetConfig:
    dates = pd.bdate_range("2022-01-03", periods=120)
    return DatasetConfig(
        feature_path=tmp_path / "features.parquet",
        output_path=tmp_path / "panel.parquet",
        scaler_path=tmp_path / "metadata" / "scaler.json",
        metadata_dir=tmp_path / "metadata",
        sequence_length=8,
        horizons=(5, 10),
        train_end=dates[59].date().isoformat(),
        validation_end=dates[89].date().isoformat(),
        batch_size=16,
        feature_columns=("feature_a", "feature_b"),
    )


def make_features() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=120)
    parts = []
    for index, ticker in enumerate(("AAA", "BBB", "CCC")):
        time = np.arange(len(dates), dtype=float)
        price = 100 + index * 5 + time * (0.1 + index * 0.02) + np.sin(time / 6)
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "adj_close": price,
                    "feature_a": np.sin(time / 6),
                    "feature_b": np.cos(time / 9 + index),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def test_transformer_experiment_saves_artifacts(tmp_path: Path) -> None:
    dataset_config = make_dataset_config(tmp_path)
    _, datasets = DatasetBuilder(dataset_config).run(make_features())
    config = TransformerExperimentConfig(
        dataset_config=tmp_path / "unused.yaml",
        artifacts_dir=tmp_path / "transformer",
        random_seed=5,
        target_horizon=5,
        model=TransformerModelConfig(
            d_model=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            dropout=0.0,
        ),
        training=TrainingConfig(
            learning_rate=0.005,
            weight_decay=0.0,
            epochs=2,
            patience=2,
            batch_size=32,
            gradient_clip=1.0,
        ),
    )

    predictions, metrics, history = TransformerExperiment(
        config,
        dataset_config,
    ).run(datasets)

    assert len(history) == 2
    assert predictions["prediction"].notna().all()
    assert {"mae", "rmse", "pearson_ic", "rank_ic"}.issubset(metrics)
    assert (config.artifacts_dir / "best_model.pt").exists()
    assert (config.artifacts_dir / "training_history.csv").exists()


def test_date_grouped_sampler_keeps_cross_sections_together(tmp_path: Path) -> None:
    dataset_config = make_dataset_config(tmp_path)
    _, datasets = DatasetBuilder(dataset_config).run(make_features())
    dataset = datasets["train"]
    sampler = DateGroupedBatchSampler(dataset, shuffle=False)
    batches = list(sampler)

    assert sum(map(len, batches)) == len(dataset)
    for batch in batches:
        dates = {dataset.metadata(index).date for index in batch}
        assert len(dates) == 1


def test_transformer_experiment_can_train_all_horizons(tmp_path: Path) -> None:
    dataset_config = make_dataset_config(tmp_path)
    _, datasets = DatasetBuilder(dataset_config).run(make_features())
    config = TransformerExperimentConfig(
        dataset_config=tmp_path / "unused.yaml",
        artifacts_dir=tmp_path / "transformer_multi",
        random_seed=7,
        target_horizon=None,
        model=TransformerModelConfig(
            d_model=8,
            num_heads=2,
            num_layers=1,
            feedforward_dim=16,
            dropout=0.0,
        ),
        training=TrainingConfig(
            learning_rate=0.005,
            weight_decay=0.0,
            epochs=1,
            patience=1,
            batch_size=32,
            gradient_clip=1.0,
            correlation_loss_weight=0.2,
        ),
    )

    predictions, _, _ = TransformerExperiment(config, dataset_config).run(datasets)
    summary = pd.read_json(config.artifacts_dir / "metrics.json", typ="series")

    assert set(predictions["horizon"]) == {5, 10}
    assert predictions.groupby(["date", "ticker"]).size().nunique() == 1
    assert summary["horizons"] == [5, 10]
    assert set(summary["metrics_by_horizon"]) == {"5", "10"}
    assert summary["correlation_loss_weight"] == 0.2
