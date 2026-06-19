from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from equity_transformer.reporting.equity_comparison import (
    EquityComparisonConfig,
    EquityComparisonPipeline,
    EquityCurveSource,
    compare_equity_curves,
)


def make_curve(start: str, periods: int, daily_return: float) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    returns = np.full(periods, daily_return)
    returns[0] = 0.0
    return pd.DataFrame(
        {
            "date": dates,
            "net_return": returns,
            "nav": (1 + returns).cumprod() * 100,
            "turnover": 0.0,
            "cost": 0.0,
        }
    )


def test_equity_comparison_recomputes_metrics_on_common_period() -> None:
    comparison, aligned = compare_equity_curves(
        {
            "factor": make_curve("2024-01-01", 5, 0.01),
            "model": make_curve("2024-01-03", 3, 0.02),
        }
    )
    indexed = comparison.set_index("portfolio")

    assert indexed["common_start"].eq(pd.Timestamp("2024-01-03")).all()
    assert indexed["common_end"].eq(pd.Timestamp("2024-01-05")).all()
    assert indexed["observations"].eq(2).all()
    assert np.isclose(indexed.loc["factor", "total_return"], 1.01**2 - 1)
    assert np.isclose(indexed.loc["model", "total_return"], 1.02**2 - 1)
    assert aligned["factor"]["nav"].iloc[0] == 1.0


def test_equity_comparison_rejects_non_overlapping_curves() -> None:
    with pytest.raises(ValueError, match="overlapping period"):
        compare_equity_curves(
            {
                "old": make_curve("2024-01-01", 2, 0.01),
                "new": make_curve("2024-02-01", 2, 0.01),
            }
        )


def test_equity_comparison_pipeline_writes_artifacts(tmp_path: Path) -> None:
    config = EquityComparisonConfig(
        sources=(
            EquityCurveSource("factor", tmp_path / "factor.csv"),
            EquityCurveSource("model", tmp_path / "model.csv"),
        ),
        output_path=tmp_path / "comparison" / "common_period.csv",
        metadata_path=tmp_path / "comparison" / "manifest.json",
        annualization_factor=252,
        risk_free_rate=0.0,
    )

    comparison = EquityComparisonPipeline(config).run(
        {
            "factor": make_curve("2024-01-01", 5, 0.01),
            "model": make_curve("2024-01-03", 3, 0.02),
        }
    )

    assert len(comparison) == 2
    assert config.output_path.exists()
    assert config.metadata_path.exists()
    assert (config.output_path.parent / "aligned_curves" / "factor.csv").exists()
