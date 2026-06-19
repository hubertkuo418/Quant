from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class FactorSignalConfig:
    feature_path: Path
    selected_factors_path: Path
    output_path: Path
    metadata_path: Path
    date_column: str
    ticker_column: str
    score_column: str
    weight_column: str
    passthrough_columns: tuple[str, ...] = ()


def load_factor_signal_config(path: str | Path) -> FactorSignalConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return FactorSignalConfig(
        feature_path=Path(payload["feature_path"]),
        selected_factors_path=Path(payload["selected_factors_path"]),
        output_path=Path(payload["output_path"]),
        metadata_path=Path(payload["metadata_path"]),
        date_column=str(payload.get("date_column", "date")),
        ticker_column=str(payload.get("ticker_column", "ticker")),
        score_column=str(payload.get("score_column", "factor_score")),
        weight_column=str(payload.get("weight_column", "mean_rank_ic")),
        passthrough_columns=tuple(
            str(column) for column in payload.get("passthrough_columns", [])
        ),
    )


class FactorSignalPipeline:
    def __init__(self, config: FactorSignalConfig) -> None:
        self.config = config

    def run(
        self,
        features: pd.DataFrame | None = None,
        selected_factors: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        feature_frame = (
            features.copy()
            if features is not None
            else pd.read_parquet(self.config.feature_path)
        )
        selected = (
            selected_factors.copy()
            if selected_factors is not None
            else pd.read_csv(self.config.selected_factors_path)
        )
        signals = build_ic_weighted_factor_signal(
            feature_frame,
            selected,
            date_column=self.config.date_column,
            ticker_column=self.config.ticker_column,
            score_column=self.config.score_column,
            weight_column=self.config.weight_column,
            passthrough_columns=self.config.passthrough_columns,
        )
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        signals.to_parquet(self.config.output_path, index=False)
        self.config.metadata_path.write_text(
            json.dumps(
                {
                    "run_utc": datetime.now(UTC).isoformat(),
                    "score_column": self.config.score_column,
                    "weight_column": self.config.weight_column,
                    "passthrough_columns": list(self.config.passthrough_columns),
                    "factor_count": len(selected),
                    "rows": len(signals),
                    "dates": int(signals[self.config.date_column].nunique())
                    if not signals.empty
                    else 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return signals


def build_ic_weighted_factor_signal(
    features: pd.DataFrame,
    selected_factors: pd.DataFrame,
    date_column: str = "date",
    ticker_column: str = "ticker",
    score_column: str = "factor_score",
    weight_column: str = "mean_rank_ic",
    passthrough_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    required = {date_column, ticker_column}
    missing = required - set(features.columns)
    if missing:
        raise ValueError(f"Feature panel missing columns: {sorted(missing)}")
    if {"factor", weight_column}.difference(selected_factors.columns):
        raise ValueError("Selected factors must include factor and weight columns.")

    selected = selected_factors[["factor", weight_column]].dropna().copy()
    factor_columns = selected["factor"].tolist()
    passthrough = tuple(
        column
        for column in passthrough_columns
        if column not in {date_column, ticker_column}
    )
    missing_factors = sorted(
        set(factor_columns).union(passthrough) - set(features.columns)
    )
    if missing_factors:
        raise ValueError(f"Feature panel missing requested columns: {missing_factors}")

    weights = selected.set_index("factor")[weight_column].astype(float)
    if float(weights.abs().sum()) == 0:
        raise ValueError("Selected factor weights must not all be zero.")
    weights = weights / weights.abs().sum()

    frame_columns = list(
        dict.fromkeys([date_column, ticker_column, *factor_columns, *passthrough])
    )
    frame = features[frame_columns].copy()
    frame[date_column] = pd.to_datetime(frame[date_column])
    normalized = frame.groupby(date_column, group_keys=False)[factor_columns].apply(
        _cross_sectional_zscore
    )
    scores = normalized.mul(weights, axis=1).sum(axis=1)
    return (
        frame[[date_column, ticker_column, *passthrough]]
        .assign(**{score_column: scores})
        .dropna(subset=[score_column])
        .sort_values([date_column, ticker_column])
        .reset_index(drop=True)
    )


def _cross_sectional_zscore(frame: pd.DataFrame) -> pd.DataFrame:
    centered = frame - frame.mean()
    scale = frame.std(ddof=0)
    scale = scale.mask(scale.eq(0))
    return centered.div(scale).fillna(0.0)
