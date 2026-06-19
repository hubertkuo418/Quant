from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


@dataclass(frozen=True)
class FactorPanelConfig:
    feature_path: Path
    alpha_path: Path
    output_path: Path
    metadata_path: Path


def load_factor_panel_config(path: str | Path) -> FactorPanelConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return FactorPanelConfig(
        feature_path=Path(payload["feature_path"]),
        alpha_path=Path(payload["alpha_path"]),
        output_path=Path(payload["output_path"]),
        metadata_path=Path(payload["metadata_path"]),
    )


def merge_feature_alpha_panels(
    features: pd.DataFrame,
    alphas: pd.DataFrame,
) -> pd.DataFrame:
    _validate_keys(features, "features")
    _validate_keys(alphas, "alphas")
    overlap = set(features.columns).intersection(alphas.columns) - {"date", "ticker"}
    if overlap:
        raise ValueError(f"Overlapping factor columns found: {sorted(overlap)}")
    merged = features.copy()
    merged["date"] = pd.to_datetime(merged["date"])
    alpha_frame = alphas.copy()
    alpha_frame["date"] = pd.to_datetime(alpha_frame["date"])
    return (
        merged.merge(alpha_frame, on=["date", "ticker"], how="left")
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


class FactorPanelPipeline:
    def __init__(self, config: FactorPanelConfig) -> None:
        self.config = config

    def run(
        self,
        features: pd.DataFrame | None = None,
        alphas: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        feature_frame = (
            features.copy()
            if features is not None
            else pd.read_parquet(self.config.feature_path)
        )
        alpha_frame = (
            alphas.copy()
            if alphas is not None
            else pd.read_parquet(self.config.alpha_path)
        )
        merged = merge_feature_alpha_panels(feature_frame, alpha_frame)
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        merged.to_parquet(self.config.output_path, index=False)
        self.config.metadata_path.write_text(
            json.dumps(
                {
                    "run_utc": datetime.now(UTC).isoformat(),
                    "feature_rows": len(feature_frame),
                    "alpha_rows": len(alpha_frame),
                    "output_rows": len(merged),
                    "output_columns": list(merged.columns),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return merged


def _validate_keys(frame: pd.DataFrame, name: str) -> None:
    missing = {"date", "ticker"}.difference(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required keys: {sorted(missing)}")
    if frame[["date", "ticker"]].duplicated().any():
        raise ValueError(f"{name} has duplicate (date, ticker) rows.")
