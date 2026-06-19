from __future__ import annotations

import copy
import json
import random
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from equity_transformer.baselines.metrics import regression_metrics
from equity_transformer.datasets.builder import DatasetBuilder
from equity_transformer.datasets.config import (
    DatasetConfig,
    load_dataset_config,
)
from equity_transformer.datasets.sequence import EquitySequenceDataset
from equity_transformer.models.recurrent import RecurrentSequenceModel
from equity_transformer.training.recurrent_config import RecurrentExperimentConfig


class RecurrentExperiment:
    def __init__(
        self,
        config: RecurrentExperimentConfig,
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        self.config = config
        self.dataset_config = dataset_config or load_dataset_config(
            config.dataset_config
        )

    def run(
        self,
        datasets: dict[str, EquitySequenceDataset] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if datasets is None:
            _, datasets = DatasetBuilder(self.dataset_config).run()
        self._set_seed()
        model = self._create_model()
        history = self._fit(model, datasets["train"], datasets["validation"])
        predictions = self._predict(model, datasets["test"])
        metrics = self._metric_frame(predictions)
        self._save(model, predictions, metrics, history)
        return predictions, metrics, history

    def _create_model(self) -> RecurrentSequenceModel:
        model = self.config.model
        return RecurrentSequenceModel(
            feature_dim=len(self.dataset_config.feature_columns),
            output_dim=len(self.dataset_config.horizons),
            model_type=self.config.model_type,
            hidden_dim=model.hidden_dim,
            num_layers=model.num_layers,
            dropout=model.dropout,
        )

    def _fit(
        self,
        model: RecurrentSequenceModel,
        train: EquitySequenceDataset,
        validation: EquitySequenceDataset,
    ) -> pd.DataFrame:
        if not train or not validation:
            raise ValueError("Train and validation datasets must be non-empty.")
        training = self.config.training
        train_loader = DataLoader(train, batch_size=training.batch_size, shuffle=True)
        validation_loader = DataLoader(
            validation, batch_size=training.batch_size, shuffle=False
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=training.learning_rate,
            weight_decay=training.weight_decay,
        )
        loss_function = nn.HuberLoss()
        best_state = copy.deepcopy(model.state_dict())
        best_validation = float("inf")
        stale_epochs = 0
        rows = []
        for epoch in range(1, training.epochs + 1):
            train_loss = self._train_epoch(
                model, train_loader, optimizer, loss_function
            )
            validation_loss = self._evaluate_loss(
                model, validation_loader, loss_function
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
        model: RecurrentSequenceModel,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        loss_function: nn.Module,
    ) -> float:
        model.train()
        total_loss = 0.0
        total_rows = 0
        for features, targets in loader:
            optimizer.zero_grad()
            loss = loss_function(model(features), targets)
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
        model: RecurrentSequenceModel,
        loader: DataLoader,
        loss_function: nn.Module,
    ) -> float:
        model.eval()
        total_loss = 0.0
        total_rows = 0
        with torch.no_grad():
            for features, targets in loader:
                loss = loss_function(model(features), targets)
                total_loss += loss.item() * len(features)
                total_rows += len(features)
        return total_loss / total_rows

    def _predict(
        self,
        model: RecurrentSequenceModel,
        dataset: EquitySequenceDataset,
    ) -> pd.DataFrame:
        features, targets = dataset.as_arrays()
        model.eval()
        with torch.no_grad():
            predictions = model(torch.from_numpy(features)).numpy()
        rows = []
        for sample_index, sample in enumerate(dataset.sample_metadata):
            for horizon_index, horizon in enumerate(self.dataset_config.horizons):
                rows.append(
                    {
                        "model": self.config.model_type,
                        "split": "test",
                        "date": sample.date,
                        "ticker": sample.ticker,
                        "horizon": horizon,
                        "prediction": float(predictions[sample_index, horizon_index]),
                        "target": float(targets[sample_index, horizon_index]),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for horizon, group in predictions.groupby("horizon"):
            rows.append({"horizon": horizon, **regression_metrics(group)})
        return pd.DataFrame(rows).sort_values("horizon")

    def _save(
        self,
        model: RecurrentSequenceModel,
        predictions: pd.DataFrame,
        metrics: pd.DataFrame,
        history: pd.DataFrame,
    ) -> None:
        output = self.config.artifacts_dir
        output.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), output / "best_model.pt")
        predictions.to_parquet(output / "test_predictions.parquet", index=False)
        metrics.to_csv(output / "metrics.csv", index=False)
        history.to_csv(output / "training_history.csv", index=False)
        (output / "run.json").write_text(
            json.dumps(
                {
                    "run_utc": datetime.now(UTC).isoformat(),
                    "model_type": self.config.model_type,
                    "epochs_completed": len(history),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _set_seed(self) -> None:
        random.seed(self.config.random_seed)
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)
