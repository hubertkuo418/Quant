from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.factors.signals import (
    FactorSignalConfig,
    FactorSignalPipeline,
    build_ic_weighted_factor_signal,
)


def make_features() -> pd.DataFrame:
    dates = pd.to_datetime(["2024-01-01"] * 3 + ["2024-01-02"] * 3)
    tickers = ["AAA", "BBB", "CCC"] * 2
    return pd.DataFrame(
        {
            "date": dates,
            "ticker": tickers,
            "momentum": [1.0, 2.0, 3.0, 2.0, 3.0, 4.0],
            "volatility": [3.0, 2.0, 1.0, 4.0, 3.0, 2.0],
            "sector": ["tech", "tech", "defensive", "tech", "tech", "defensive"],
        }
    )


def make_selected() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor": ["momentum", "volatility"],
            "mean_rank_ic": [0.08, -0.04],
        }
    )


def test_build_ic_weighted_factor_signal_scores_cross_sectionally() -> None:
    signals = build_ic_weighted_factor_signal(make_features(), make_selected())

    first_day = signals[signals["date"] == pd.Timestamp("2024-01-01")]
    assert first_day.sort_values("factor_score")["ticker"].tolist() == [
        "AAA",
        "BBB",
        "CCC",
    ]
    assert signals["factor_score"].notna().all()


def test_build_ic_weighted_factor_signal_rejects_missing_factor() -> None:
    selected = pd.DataFrame({"factor": ["missing"], "mean_rank_ic": [0.1]})

    with pytest.raises(ValueError, match="missing requested columns"):
        build_ic_weighted_factor_signal(make_features(), selected)


def test_build_ic_weighted_factor_signal_preserves_passthrough_columns() -> None:
    signals = build_ic_weighted_factor_signal(
        make_features(),
        make_selected(),
        passthrough_columns=("sector", "volatility"),
    )

    assert {"sector", "volatility", "factor_score"}.issubset(signals.columns)
    assert signals.loc[0, "sector"] == "tech"


def test_factor_signal_pipeline_writes_outputs(tmp_path: Path) -> None:
    config = FactorSignalConfig(
        feature_path=tmp_path / "features.parquet",
        selected_factors_path=tmp_path / "selected.csv",
        output_path=tmp_path / "factor_signals.parquet",
        metadata_path=tmp_path / "factor_signals.json",
        date_column="date",
        ticker_column="ticker",
        score_column="factor_score",
        weight_column="mean_rank_ic",
        passthrough_columns=("sector",),
    )
    signals = FactorSignalPipeline(config).run(make_features(), make_selected())

    assert len(signals) == 6
    assert config.output_path.exists()
    assert config.metadata_path.exists()
