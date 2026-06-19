from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from equity_transformer.datasets.config import DatasetConfig
from equity_transformer.datasets.scaling import FeatureScaler
from equity_transformer.datasets.sequence import EquitySequenceDataset
from equity_transformer.datasets.targets import (
    add_forward_return_targets,
    assign_purged_splits,
)


class DatasetBuilder:
    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def run(
        self, features: pd.DataFrame | None = None
    ) -> tuple[pd.DataFrame, dict[str, EquitySequenceDataset]]:
        frame = (
            features.copy()
            if features is not None
            else pd.read_parquet(self.config.feature_path)
        )
        frame["date"] = pd.to_datetime(frame["date"])
        targeted = add_forward_return_targets(frame, self.config.horizons)
        targeted = assign_purged_splits(
            targeted,
            self.config.horizons,
            self.config.train_end,
            self.config.validation_end,
        )

        scaler_fit_rows = targeted["date"] <= pd.Timestamp(self.config.train_end)
        scaler = FeatureScaler.fit(
            targeted.loc[scaler_fit_rows], self.config.feature_columns
        )
        model_panel = scaler.transform(targeted)
        scaler.save(self.config.scaler_path)

        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        model_panel.to_parquet(self.config.output_path, index=False)

        datasets = {
            split: EquitySequenceDataset(
                model_panel,
                self.config.feature_columns,
                self.config.target_columns,
                self.config.sequence_length,
                split,
            )
            for split in ("train", "validation", "test")
        }
        self._write_manifest(model_panel, datasets)
        return model_panel, datasets

    def _write_manifest(
        self,
        panel: pd.DataFrame,
        datasets: dict[str, EquitySequenceDataset],
    ) -> None:
        run_time = datetime.now(UTC)
        payload = {
            "run_utc": run_time.isoformat(),
            "sequence_length": self.config.sequence_length,
            "feature_columns": list(self.config.feature_columns),
            "target_columns": list(self.config.target_columns),
            "train_end": self.config.train_end,
            "validation_end": self.config.validation_end,
            "panel_rows": len(panel),
            "sample_counts": {
                split: len(dataset) for split, dataset in datasets.items()
            },
        }
        path = self.config.metadata_dir / f"dataset_{run_time:%Y%m%dT%H%M%SZ}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
