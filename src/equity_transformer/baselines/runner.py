from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from equity_transformer.baselines.config import BaselineConfig
from equity_transformer.baselines.metrics import regression_metrics
from equity_transformer.baselines.models import (
    fit_historical_mean,
    fit_lightgbm,
    fit_mlp,
    fit_momentum,
    fit_random_forest,
    fit_ridge,
    fit_xgboost,
)
from equity_transformer.datasets.builder import DatasetBuilder
from equity_transformer.datasets.config import (
    DatasetConfig,
    load_dataset_config,
)
from equity_transformer.datasets.sequence import EquitySequenceDataset


class BaselineExperiment:
    def __init__(
        self,
        config: BaselineConfig,
        dataset_config: DatasetConfig | None = None,
    ) -> None:
        self.config = config
        self.dataset_config = dataset_config or load_dataset_config(
            config.dataset_config
        )

    def run(
        self,
        datasets: dict[str, EquitySequenceDataset] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if datasets is None:
            _, datasets = DatasetBuilder(self.dataset_config).run()
        arrays = {
            split: dataset.as_arrays() for split, dataset in datasets.items()
        }
        x_train, y_train = arrays["train"]
        x_validation, y_validation = arrays["validation"]
        x_test, y_test = arrays["test"]
        if min(len(x_train), len(x_validation), len(x_test)) == 0:
            raise ValueError("Train, validation, and test datasets must be non-empty.")

        x_train_last = x_train[:, -1, :]
        x_validation_last = x_validation[:, -1, :]
        x_test_last = x_test[:, -1, :]
        momentum_index = self.dataset_config.feature_columns.index(
            self.config.momentum_feature
        )

        model_predictions = {
            "historical_mean": fit_historical_mean(y_train, len(x_test)),
            "momentum": fit_momentum(
                x_train_last,
                y_train,
                x_test_last,
                momentum_index,
                self.config.ridge_alpha,
            ),
            "ridge": fit_ridge(
                x_train_last,
                y_train,
                x_test_last,
                self.config.ridge_alpha,
            ),
            "random_forest": fit_random_forest(
                x_train_last,
                y_train,
                x_test_last,
                self.config,
            ),
            "xgboost": fit_xgboost(
                x_train_last,
                y_train,
                x_validation_last,
                y_validation,
                x_test_last,
                self.config,
            ),
            "lightgbm": fit_lightgbm(
                x_train_last,
                y_train,
                x_test_last,
                self.config,
            ),
            "mlp": fit_mlp(
                x_train,
                y_train,
                x_validation,
                y_validation,
                x_test,
                self.config.mlp,
                self.config.random_seed,
            ),
        }
        predictions = self._prediction_frame(
            model_predictions,
            y_test,
            datasets["test"],
        )
        metrics = self._metric_frame(predictions)
        self._save_artifacts(predictions, metrics)
        return predictions, metrics

    def _prediction_frame(
        self,
        model_predictions: dict[str, np.ndarray],
        targets: np.ndarray,
        test_dataset: EquitySequenceDataset,
    ) -> pd.DataFrame:
        rows = []
        metadata = test_dataset.sample_metadata
        for model_name, prediction_matrix in model_predictions.items():
            for sample_index, sample in enumerate(metadata):
                for horizon_index, horizon in enumerate(self.dataset_config.horizons):
                    rows.append(
                        {
                            "model": model_name,
                            "split": "test",
                            "date": sample.date,
                            "ticker": sample.ticker,
                            "horizon": horizon,
                            "prediction": float(
                                prediction_matrix[sample_index, horizon_index]
                            ),
                            "target": float(targets[sample_index, horizon_index]),
                        }
                    )
        return pd.DataFrame(rows).sort_values(
            ["model", "horizon", "date", "ticker"]
        )

    @staticmethod
    def _metric_frame(predictions: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for (model, horizon), group in predictions.groupby(["model", "horizon"]):
            rows.append(
                {
                    "model": model,
                    "horizon": horizon,
                    **regression_metrics(group),
                }
            )
        return pd.DataFrame(rows).sort_values(["horizon", "rmse", "model"])

    def _save_artifacts(
        self,
        predictions: pd.DataFrame,
        metrics: pd.DataFrame,
    ) -> None:
        self.config.artifacts_dir.mkdir(parents=True, exist_ok=True)
        predictions.to_parquet(
            self.config.artifacts_dir / "test_predictions.parquet", index=False
        )
        metrics.to_csv(self.config.artifacts_dir / "metrics.csv", index=False)
        summary = {
            "run_utc": datetime.now(UTC).isoformat(),
            "seed": self.config.random_seed,
            "models": sorted(predictions["model"].unique().tolist()),
            "horizons": list(self.dataset_config.horizons),
            "test_samples": int(
                predictions[["date", "ticker"]].drop_duplicates().shape[0]
            ),
        }
        (self.config.artifacts_dir / "run.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
