from __future__ import annotations

import copy
import json
import random
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler

from equity_transformer.baselines.metrics import regression_metrics
from equity_transformer.datasets.builder import DatasetBuilder
from equity_transformer.datasets.config import (
    DatasetConfig,
    load_dataset_config,
)
from equity_transformer.datasets.sequence import EquitySequenceDataset
from equity_transformer.models.transformer import TimeSeriesTransformer
from equity_transformer.training.config import (
    TrainingConfig,
    TransformerExperimentConfig,
)


class TransformerExperiment:
    def __init__(
        self,
        config: TransformerExperimentConfig,
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        self.config = config
        self.dataset_config = dataset_config or load_dataset_config(
            config.dataset_config
        )

    def run(
        self,
        datasets: dict[str, EquitySequenceDataset] | None = None,
    ) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
        if datasets is None:
            _, datasets = DatasetBuilder(self.dataset_config).run()
        self._set_seed()
        model = self._create_model()
        history = self._fit(model, datasets["train"], datasets["validation"])
        predictions = self._predict(model, datasets["test"])
        metrics = regression_metrics(predictions)
        self._save(model, predictions, metrics, history)
        return predictions, metrics, history

    def _create_model(self) -> TimeSeriesTransformer:
        model = self.config.model
        return TimeSeriesTransformer(
            feature_dim=len(self.dataset_config.feature_columns),
            sequence_length=self.dataset_config.sequence_length,
            output_dim=len(self._target_indices()),
            d_model=model.d_model,
            num_heads=model.num_heads,
            num_layers=model.num_layers,
            feedforward_dim=model.feedforward_dim,
            dropout=model.dropout,
        )

    def _fit(
        self,
        model: TimeSeriesTransformer,
        train: EquitySequenceDataset,
        validation: EquitySequenceDataset,
    ) -> pd.DataFrame:
        if not train or not validation:
            raise ValueError("Train and validation datasets must be non-empty.")
        target_indices = self._target_indices()
        training = self.config.training
        train_loader = _make_training_loader(train, training, shuffle=True)
        validation_loader = _make_training_loader(
            validation, training, shuffle=False
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )
        loss_function = CorrelationAwareLoss(training.correlation_loss_weight)
        best_state = copy.deepcopy(model.state_dict())
        best_validation = float("inf")
        stale_epochs = 0
        rows = []

        for epoch in range(1, training.epochs + 1):
            train_loss = self._train_epoch(
                model,
                train_loader,
                target_indices,
                optimizer,
                loss_function,
            )
            validation_loss = self._evaluate_loss(
                model,
                validation_loader,
                target_indices,
                loss_function,
            )
            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                }
            )
            if validation_loss < best_validation:
                best_validation = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= training.patience:
                    break

        model.load_state_dict(best_state)
        return pd.DataFrame(rows)

    def _train_epoch(
        self,
        model: TimeSeriesTransformer,
        loader: DataLoader,
        target_indices: list[int],
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, targets in loader:
            optimizer.zero_grad()
            prediction = model(features)
            loss = loss_function(prediction, targets[:, target_indices])
            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), self.config.training.gradient_clip
            )
            optimizer.step()
            total_loss += loss.item() * len(features)
            total_rows += len(features)
        return total_loss / total_rows

    @staticmethod
    def _evaluate_loss(
        model: TimeSeriesTransformer,
        loader: DataLoader,
        target_indices: list[int],
        loss_function: nn.Module,
    ) -> float:
        model.eval()
        total_loss = 0.0
        total_rows = 0
        with torch.no_grad():
            for features, targets in loader:
                loss = loss_function(
                    model(features),
                    targets[:, target_indices],
                )
                total_loss += loss.item() * len(features)
                total_rows += len(features)
        return total_loss / total_rows

    def _predict(
        self,
        model: TimeSeriesTransformer,
        dataset: EquitySequenceDataset,
    ) -> pd.DataFrame:
        features, targets = dataset.as_arrays()
        target_indices = self._target_indices()
        horizons = self._output_horizons()
        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(features)).numpy()

        rows = []
        for sample_index, item in enumerate(dataset.sample_metadata):
            for output_index, horizon in enumerate(horizons):
                rows.append(
                    {
                        "model": "transformer_v1",
                        "split": "test",
                        "date": item.date,
                        "ticker": item.ticker,
                        "horizon": horizon,
                        "prediction": float(predictions[sample_index, output_index]),
                        "target": float(
                            targets[sample_index, target_indices[output_index]]
                        ),
                    }
                )
        return pd.DataFrame(rows)

    def _save(
        self,
        model: TimeSeriesTransformer,
        predictions: pd.DataFrame,
        metrics: dict[str, float],
        history: pd.DataFrame,
    ) -> None:
        output = self.config.artifacts_dir
        output.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output / "best_model.pt")
        predictions.to_parquet(output / "test_predictions.parquet", index=False)
        history.to_csv(output / "training_history.csv", index=False)
        summary = {
            "run_utc": datetime.now(UTC).isoformat(),
            "target_horizon": self.config.target_horizon,
            "horizons": self._output_horizons(),
            "seed": self.config.random_seed,
            "epochs_completed": len(history),
            "correlation_loss_weight": self.config.training.correlation_loss_weight,
            "metrics": metrics,
            "metrics_by_horizon": _metrics_by_horizon(predictions),
        }
        (output / "metrics.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True),
            encoding="utf-8",
        )

    def _set_seed(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

    def _target_indices(self) -> list[int]:
        if self.config.target_horizon is None:
            return list(range(len(self.dataset_config.horizons)))
        return [self.dataset_config.horizons.index(self.config.target_horizon)]

    def _output_horizons(self) -> list[int]:
        if self.config.target_horizon is None:
            return list(self.dataset_config.horizons)
        return [self.config.target_horizon]


def _metrics_by_horizon(predictions: pd.DataFrame) -> dict[str, dict[str, float]]:
    return {
        str(horizon): regression_metrics(group)
        for horizon, group in predictions.groupby("horizon", sort=True)
    }


class DateGroupedBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset: EquitySequenceDataset, shuffle: bool) -> None:
        groups: dict[pd.Timestamp, list[int]] = defaultdict(list)
        for index, metadata in enumerate(dataset.sample_metadata):
            groups[pd.Timestamp(metadata.date)].append(index)
        self._batches = list(groups.values())
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        if not self.shuffle:
            yield from self._batches
            return
        order = torch.randperm(len(self._batches)).tolist()
        for index in order:
            yield self._batches[index]

    def __len__(self) -> int:
        return len(self._batches)


class CorrelationAwareLoss(nn.Module):
    def __init__(self, correlation_weight: float) -> None:
        super().__init__()
        if correlation_weight < 0:
            raise ValueError("correlation_weight must be non-negative.")
        self.correlation_weight = correlation_weight
        self.base_loss = nn.HuberLoss()

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        base = self.base_loss(prediction, target)
        if self.correlation_weight == 0 or len(prediction) < 2:
            return base
        prediction_centered = prediction - prediction.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)
        numerator = (prediction_centered * target_centered).sum(dim=0)
        denominator = torch.sqrt(
            prediction_centered.square().sum(dim=0)
            * target_centered.square().sum(dim=0)
        )
        valid = denominator > torch.finfo(prediction.dtype).eps
        if not torch.any(valid):
            return base
        correlation = numerator[valid] / denominator[valid]
        return base + self.correlation_weight * (1 - correlation.mean())


def _make_training_loader(
    dataset: EquitySequenceDataset,
    training: TrainingConfig,
    shuffle: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    if training.correlation_loss_weight > 0:
        return DataLoader(
            dataset,
            batch_sampler=DateGroupedBatchSampler(dataset, shuffle),
        )
    return DataLoader(
        dataset,
        batch_size=training.batch_size,
        shuffle=shuffle,
    )
