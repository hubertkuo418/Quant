from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class FeatureScaler:
    columns: tuple[str, ...]
    medians: dict[str, float]
    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: tuple[str, ...]) -> FeatureScaler:
        missing = set(columns).difference(frame.columns)
        if missing:
            raise ValueError(f"Missing configured features: {sorted(missing)}")

        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for column in columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            median = float(values.median()) if values.notna().any() else 0.0
            filled = values.fillna(median)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            medians[column] = median
            means[column] = mean
            scales[column] = scale if np.isfinite(scale) and scale > 0 else 1.0
        return cls(columns, medians, means, scales)

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in self.columns:
            values = pd.to_numeric(result[column], errors="coerce")
            result[column] = (
                values.fillna(self.medians[column]) - self.means[column]
            ) / self.scales[column]
        return result

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "columns": list(self.columns),
            "medians": self.medians,
            "means": self.means,
            "scales": self.scales,
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
