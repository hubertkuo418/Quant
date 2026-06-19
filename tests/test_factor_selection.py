from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.factors.selection import (
    FactorSelectionConfig,
    FactorSelectionPipeline,
    select_factors,
)


def make_config(tmp_path: Path) -> FactorSelectionConfig:
    return FactorSelectionConfig(
        ic_summary_path=tmp_path / "ic.csv",
        coverage_path=tmp_path / "coverage.csv",
        output_csv_path=tmp_path / "selected.csv",
        output_json_path=tmp_path / "selected.json",
        min_coverage=0.8,
        min_periods=10,
        min_abs_rank_ic=0.02,
        min_positive_ic_rate=0.55,
        top_n=2,
    )


def make_ic_summary() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor": ["good", "weak", "sparse"],
            "mean_rank_ic": [0.08, 0.01, 0.2],
            "periods": [20, 20, 5],
            "positive_ic_rate": [0.7, 0.8, 0.9],
        }
    )


def make_coverage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor": ["good", "weak", "sparse"],
            "coverage": [0.9, 0.95, 1.0],
        }
    )


def test_select_factors_filters_and_scores(tmp_path: Path) -> None:
    selected = select_factors(make_ic_summary(), make_coverage(), make_config(tmp_path))

    assert selected["factor"].tolist() == ["good"]
    assert selected.loc[0, "selection_score"] > 0


def test_factor_selection_pipeline_writes_outputs(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    selected = FactorSelectionPipeline(config).run(make_ic_summary(), make_coverage())

    assert len(selected) == 1
    assert config.output_csv_path.exists()
    assert config.output_json_path.exists()


def test_select_factors_rejects_missing_columns(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="IC summary missing"):
        select_factors(
            pd.DataFrame({"factor": ["x"]}),
            make_coverage(),
            make_config(tmp_path),
        )
