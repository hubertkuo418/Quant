from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.baselines.config import (
    BaselineConfig,
    LightGBMConfig,
    MLPConfig,
    RandomForestConfig,
    XGBoostConfig,
)
from equity_transformer.baselines.metrics import regression_metrics
from equity_transformer.baselines.runner import BaselineExperiment
from equity_transformer.datasets.builder import DatasetBuilder
from equity_transformer.datasets.config import DatasetConfig


def make_dataset_config(tmp_path: Path) -> DatasetConfig:
    dates = pd.bdate_range("2022-01-03", periods=180)
    return DatasetConfig(
        feature_path=tmp_path / "features.parquet",
        output_path=tmp_path / "model_panel.parquet",
        scaler_path=tmp_path / "metadata" / "scaler.json",
        metadata_dir=tmp_path / "metadata",
        sequence_length=10,
        horizons=(5, 10),
        train_end=dates[89].date().isoformat(),
        validation_end=dates[134].date().isoformat(),
        batch_size=16,
        feature_columns=("return_20d", "feature_b"),
    )


def make_features(rows: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=rows)
    parts = []
    for index, ticker in enumerate(("AAA", "BBB", "CCC", "DDD")):
        trend = 0.08 + index * 0.03
        cycle = np.sin(np.arange(rows) / 8 + index)
        price = 100 + index * 10 + trend * np.arange(rows) + cycle
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "adj_close": price,
                    "return_20d": pd.Series(price).pct_change(20).to_numpy(),
                    "feature_b": cycle,
                }
            )
        )
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def make_baseline_config(tmp_path: Path) -> BaselineConfig:
    return BaselineConfig(
        dataset_config=tmp_path / "unused.yaml",
        artifacts_dir=tmp_path / "artifacts",
        random_seed=7,
        ridge_alpha=1.0,
        momentum_feature="return_20d",
        random_forest=RandomForestConfig(
            n_estimators=10,
            max_depth=2,
            min_samples_leaf=1,
        ),
        xgboost=XGBoostConfig(
            n_estimators=10,
            max_depth=2,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
        ),
        lightgbm=LightGBMConfig(
            n_estimators=10,
            max_depth=2,
            learning_rate=0.1,
            num_leaves=7,
            min_child_samples=2,
        ),
        mlp=MLPConfig(
            hidden_dims=(16,),
            dropout=0.0,
            learning_rate=0.005,
            weight_decay=0.0,
            epochs=3,
            patience=2,
            batch_size=32,
        ),
    )


def test_metrics_compute_daily_cross_sectional_ic() -> None:
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-02"] * 3 + ["2024-01-03"] * 3
            ),
            "prediction": [1, 2, 3, 3, 2, 1],
            "target": [2, 4, 6, 6, 4, 2],
        }
    )

    metrics = regression_metrics(predictions)

    assert np.isclose(metrics["pearson_ic"], 1.0)
    assert np.isclose(metrics["rank_ic"], 1.0)
    assert metrics["directional_accuracy"] == 1.0


def test_baseline_experiment_outputs_all_models(tmp_path: Path) -> None:
    dataset_config = make_dataset_config(tmp_path)
    _, datasets = DatasetBuilder(dataset_config).run(make_features())
    experiment = BaselineExperiment(
        make_baseline_config(tmp_path),
        dataset_config=dataset_config,
    )

    predictions, metrics = experiment.run(datasets)

    expected_models = {
        "historical_mean",
        "lightgbm",
        "momentum",
        "random_forest",
        "ridge",
        "xgboost",
        "mlp",
    }
    assert set(predictions["model"]) == expected_models
    assert set(metrics["model"]) == expected_models
    assert len(metrics) == len(expected_models) * len(dataset_config.horizons)
    assert predictions["prediction"].notna().all()
    assert (experiment.config.artifacts_dir / "metrics.csv").exists()
    assert (experiment.config.artifacts_dir / "test_predictions.parquet").exists()
