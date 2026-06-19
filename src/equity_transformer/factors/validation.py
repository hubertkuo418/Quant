from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from equity_transformer.datasets.targets import add_forward_return_targets
from equity_transformer.factors.config import FactorValidationConfig
from equity_transformer.factors.metrics import (
    daily_information_coefficients,
    factor_coverage,
    quantile_forward_returns,
    summarize_information_coefficients,
    summarize_quantile_returns,
)
from equity_transformer.factors.registry import FactorSpec, infer_factor_specs


class FactorValidationPipeline:
    def __init__(self, config: FactorValidationConfig) -> None:
        self.config = config

    def run(self, features: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
        frame = (
            features.copy()
            if features is not None
            else pd.read_parquet(self.config.feature_path)
        )
        frame["date"] = pd.to_datetime(frame["date"])
        target_column = f"target_{self.config.target_horizon}d"
        target_date_column = f"target_date_{self.config.target_horizon}d"
        if target_column not in frame.columns:
            frame = add_forward_return_targets(frame, (self.config.target_horizon,))
        elif target_date_column not in frame.columns:
            ordered = frame.sort_values(["ticker", "date"]).copy()
            ordered[target_date_column] = ordered.groupby("ticker")["date"].shift(
                -self.config.target_horizon
            )
            frame = ordered.sort_values(["date", "ticker"]).reset_index(drop=True)

        if self.config.start_date:
            frame = frame[frame["date"] >= pd.Timestamp(self.config.start_date)]
        if self.config.end_date:
            frame = frame[
                frame[target_date_column] <= pd.Timestamp(self.config.end_date)
            ]
        if frame.empty:
            raise ValueError("No factor observations remain in the validation period.")

        specs = self._factor_specs(frame)
        coverage = factor_coverage(frame, specs)
        daily_ic = daily_information_coefficients(
            frame,
            specs,
            target_column,
            self.config.min_cross_section,
        )
        ic_summary = summarize_information_coefficients(daily_ic)
        quantile_returns = quantile_forward_returns(
            frame,
            specs,
            target_column,
            self.config.quantiles,
            self.config.min_cross_section,
        )
        quantile_summary = summarize_quantile_returns(quantile_returns)

        outputs = {
            "coverage": coverage,
            "daily_ic": daily_ic,
            "ic_summary": ic_summary,
            "quantile_returns": quantile_returns,
            "quantile_summary": quantile_summary,
        }
        self._save(outputs, specs, target_column)
        return outputs

    def _factor_specs(self, frame: pd.DataFrame) -> list[FactorSpec]:
        columns = self.config.factor_columns or tuple(frame.columns)
        specs = infer_factor_specs(columns)
        missing = [spec.name for spec in specs if spec.name not in frame.columns]
        if missing:
            raise ValueError(f"Configured factors not found: {missing}")
        if not specs:
            raise ValueError("No factor columns were found for validation.")
        return specs

    def _save(
        self,
        outputs: dict[str, pd.DataFrame],
        specs: list[FactorSpec],
        target_column: str,
    ) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in outputs.items():
            frame.to_csv(self.config.output_dir / f"{name}.csv", index=False)
        manifest = {
            "run_utc": datetime.now(UTC).isoformat(),
            "target_column": target_column,
            "quantiles": self.config.quantiles,
            "min_cross_section": self.config.min_cross_section,
            "start_date": self.config.start_date,
            "end_date": self.config.end_date,
            "factors": [
                {
                    "name": spec.name,
                    "family": spec.family,
                    "direction": spec.direction,
                    "description": spec.description,
                }
                for spec in specs
            ],
        }
        (self.config.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
