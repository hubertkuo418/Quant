from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class FactorSelectionConfig:
    ic_summary_path: Path
    coverage_path: Path
    output_csv_path: Path
    output_json_path: Path
    min_coverage: float
    min_periods: int
    min_abs_rank_ic: float
    min_positive_ic_rate: float
    top_n: int


def load_factor_selection_config(path: str | Path) -> FactorSelectionConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return FactorSelectionConfig(
        ic_summary_path=Path(payload["ic_summary_path"]),
        coverage_path=Path(payload["coverage_path"]),
        output_csv_path=Path(payload["output_csv_path"]),
        output_json_path=Path(payload["output_json_path"]),
        min_coverage=float(payload["min_coverage"]),
        min_periods=int(payload["min_periods"]),
        min_abs_rank_ic=float(payload["min_abs_rank_ic"]),
        min_positive_ic_rate=float(payload["min_positive_ic_rate"]),
        top_n=int(payload["top_n"]),
    )


class FactorSelectionPipeline:
    def __init__(self, config: FactorSelectionConfig) -> None:
        self.config = config

    def run(
        self,
        ic_summary: pd.DataFrame | None = None,
        coverage: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        ic_frame = (
            ic_summary.copy()
            if ic_summary is not None
            else pd.read_csv(self.config.ic_summary_path)
        )
        coverage_frame = (
            coverage.copy()
            if coverage is not None
            else pd.read_csv(self.config.coverage_path)
        )
        selected = select_factors(ic_frame, coverage_frame, self.config)
        self.config.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.output_json_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(self.config.output_csv_path, index=False)
        self.config.output_json_path.write_text(
            json.dumps(
                {
                    "selected_factors": selected["factor"].tolist(),
                    "rows": len(selected),
                    "criteria": {
                        "min_coverage": self.config.min_coverage,
                        "min_periods": self.config.min_periods,
                        "min_abs_rank_ic": self.config.min_abs_rank_ic,
                        "min_positive_ic_rate": self.config.min_positive_ic_rate,
                        "top_n": self.config.top_n,
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return selected


def select_factors(
    ic_summary: pd.DataFrame,
    coverage: pd.DataFrame,
    config: FactorSelectionConfig,
) -> pd.DataFrame:
    required_ic = {"factor", "mean_rank_ic", "periods", "positive_ic_rate"}
    required_coverage = {"factor", "coverage"}
    missing_ic = required_ic.difference(ic_summary.columns)
    missing_coverage = required_coverage.difference(coverage.columns)
    if missing_ic:
        raise ValueError(f"IC summary missing columns: {sorted(missing_ic)}")
    if missing_coverage:
        raise ValueError(
            f"Coverage summary missing columns: {sorted(missing_coverage)}"
        )

    merged = ic_summary.merge(
        coverage[["factor", "coverage"]],
        on="factor",
        how="left",
    )
    filtered = merged[
        (merged["coverage"] >= config.min_coverage)
        & (merged["periods"] >= config.min_periods)
        & (merged["mean_rank_ic"].abs() >= config.min_abs_rank_ic)
        & (merged["positive_ic_rate"] >= config.min_positive_ic_rate)
    ].copy()
    filtered["selection_score"] = (
        filtered["mean_rank_ic"].abs()
        * filtered["coverage"]
        * filtered["positive_ic_rate"]
    )
    return (
        filtered.sort_values("selection_score", ascending=False)
        .head(config.top_n)
        .reset_index(drop=True)
    )
