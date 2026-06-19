from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.factors.panel import (
    FactorPanelConfig,
    FactorPanelPipeline,
    merge_feature_alpha_panels,
)


def make_features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "ticker": ["AAA", "BBB", "AAA"],
            "return_20d": [0.1, 0.2, 0.3],
        }
    )


def make_alphas() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
            "ticker": ["AAA", "BBB", "AAA"],
            "alpha_momentum_20d": [1.0, 2.0, 3.0],
        }
    )


def make_config(tmp_path: Path) -> FactorPanelConfig:
    return FactorPanelConfig(
        feature_path=tmp_path / "features.parquet",
        alpha_path=tmp_path / "alphas.parquet",
        output_path=tmp_path / "factor_panel.parquet",
        metadata_path=tmp_path / "manifest.json",
    )


def test_merge_feature_alpha_panels_preserves_keys() -> None:
    merged = merge_feature_alpha_panels(make_features(), make_alphas())

    assert len(merged) == 3
    assert {"return_20d", "alpha_momentum_20d"}.issubset(merged.columns)
    assert merged[["date", "ticker"]].duplicated().sum() == 0


def test_merge_rejects_overlapping_factor_columns() -> None:
    alphas = make_alphas().rename(columns={"alpha_momentum_20d": "return_20d"})

    with pytest.raises(ValueError, match="Overlapping"):
        merge_feature_alpha_panels(make_features(), alphas)


def test_factor_panel_pipeline_writes_output(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    merged = FactorPanelPipeline(config).run(make_features(), make_alphas())

    assert len(merged) == 3
    assert config.output_path.exists()
    assert config.metadata_path.exists()
