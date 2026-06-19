from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.datasets.builder import DatasetBuilder
from equity_transformer.datasets.config import DatasetConfig
from equity_transformer.datasets.scaling import FeatureScaler
from equity_transformer.datasets.sequence import make_dataloader
from equity_transformer.datasets.targets import (
    add_forward_return_targets,
    assign_purged_splits,
)


def make_feature_panel(rows: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-02", periods=rows)
    parts = []
    for ticker, offset in [("AAA", 0.0), ("BBB", 20.0)]:
        price = 100.0 + offset + np.arange(rows)
        parts.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "adj_close": price,
                    "feature_a": np.arange(rows, dtype=float),
                    "feature_b": np.arange(rows, dtype=float) * 2,
                }
            )
        )
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def make_config(tmp_path: Path) -> DatasetConfig:
    dates = pd.bdate_range("2023-01-02", periods=180)
    return DatasetConfig(
        feature_path=tmp_path / "features.parquet",
        output_path=tmp_path / "model_panel.parquet",
        scaler_path=tmp_path / "metadata" / "scaler.json",
        metadata_dir=tmp_path / "metadata",
        sequence_length=20,
        horizons=(5, 20),
        train_end=dates[89].date().isoformat(),
        validation_end=dates[139].date().isoformat(),
        batch_size=8,
        feature_columns=("feature_a", "feature_b"),
    )


def test_forward_targets_align_with_future_price() -> None:
    panel = make_feature_panel(30)
    targeted = add_forward_return_targets(panel, (5,))
    aaa = targeted[targeted["ticker"] == "AAA"].reset_index(drop=True)

    assert np.isclose(aaa.loc[0, "target_5d"], 105 / 100 - 1)
    assert aaa.loc[0, "target_date_5d"] == aaa.loc[5, "date"]
    assert np.isnan(aaa.loc[29, "target_5d"])


def test_split_purges_targets_crossing_boundaries(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    targeted = add_forward_return_targets(make_feature_panel(), config.horizons)
    split = assign_purged_splits(
        targeted,
        config.horizons,
        config.train_end,
        config.validation_end,
    )

    train = split[split["split"] == "train"]
    validation = split[split["split"] == "validation"]
    assert train["target_date_20d"].max() <= pd.Timestamp(config.train_end)
    assert validation["date"].min() > pd.Timestamp(config.train_end)
    assert validation["target_date_20d"].max() <= pd.Timestamp(
        config.validation_end
    )


def test_scaler_fit_is_unchanged_by_future_values() -> None:
    panel = make_feature_panel(50)
    train = panel[panel["date"] <= panel["date"].sort_values().unique()[24]]
    original = FeatureScaler.fit(train, ("feature_a", "feature_b"))

    changed = panel.copy()
    changed.loc[changed["date"] > train["date"].max(), "feature_a"] = 1e12
    changed_train = changed[changed["date"] <= train["date"].max()]
    recalculated = FeatureScaler.fit(changed_train, ("feature_a", "feature_b"))

    assert original.means == recalculated.means
    assert original.scales == recalculated.scales


def test_builder_produces_expected_tensor_shapes(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    panel, datasets = DatasetBuilder(config).run(make_feature_panel())

    assert config.output_path.exists()
    assert config.scaler_path.exists()
    assert set(panel["split"]) == {"train", "validation", "test", "unassigned"}
    assert all(len(datasets[split]) > 0 for split in datasets)

    loader = make_dataloader(datasets["train"], batch_size=8, shuffle=False)
    features, targets = next(iter(loader))
    assert features.shape == (8, 20, 2)
    assert targets.shape == (8, 2)
    assert np.isfinite(features.numpy()).all()
