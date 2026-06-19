from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class FactorValidationConfig:
    feature_path: Path
    output_dir: Path
    target_horizon: int
    quantiles: int
    min_cross_section: int
    factor_columns: tuple[str, ...] | None
    start_date: str | None = None
    end_date: str | None = None


def load_factor_validation_config(path: str | Path) -> FactorValidationConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    configured_factors = payload.get("factor_columns")
    factor_columns = (
        tuple(str(column) for column in configured_factors)
        if configured_factors
        else None
    )
    return FactorValidationConfig(
        feature_path=Path(payload["feature_path"]),
        output_dir=Path(payload["output_dir"]),
        target_horizon=int(payload["target_horizon"]),
        quantiles=int(payload["quantiles"]),
        min_cross_section=int(payload["min_cross_section"]),
        factor_columns=factor_columns,
        start_date=str(payload["start_date"]) if payload.get("start_date") else None,
        end_date=str(payload["end_date"]) if payload.get("end_date") else None,
    )
