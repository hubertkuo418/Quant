from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class StrategyRunRecord:
    run_id: str
    path: Path
    manifest: dict[str, Any]


class StrategyRunRegistry:
    def __init__(self, root: str | Path = "artifacts/studio/runs") -> None:
        self.root = Path(root)

    def list_runs(self) -> list[StrategyRunRecord]:
        if not self.root.exists():
            return []
        records = []
        for path in sorted(self.root.iterdir(), reverse=True):
            manifest_path = path / "manifest.json"
            if path.is_dir() and manifest_path.exists():
                records.append(
                    StrategyRunRecord(
                        run_id=path.name,
                        path=path,
                        manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
                    )
                )
        return records

    def get(self, run_id: str) -> StrategyRunRecord:
        if not run_id or Path(run_id).name != run_id:
            raise ValueError("Invalid run_id.")
        path = self.root / run_id
        manifest_path = path / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Strategy run does not exist: {run_id}")
        return StrategyRunRecord(
            run_id=run_id,
            path=path,
            manifest=json.loads(manifest_path.read_text(encoding="utf-8")),
        )

    def summary(self) -> pd.DataFrame:
        rows = []
        for record in self.list_runs():
            manifest = record.manifest
            metrics = manifest.get("metrics", {})
            rows.append(
                {
                    "run_id": record.run_id,
                    "strategy": manifest.get("strategy_name"),
                    "version": manifest.get("strategy_version"),
                    "created_utc": manifest.get("created_utc"),
                    "spec_hash": manifest.get("spec_hash"),
                    "total_return": metrics.get("total_return"),
                    "annual_return": metrics.get("annual_return"),
                    "sharpe_ratio": metrics.get("sharpe_ratio"),
                    "max_drawdown": metrics.get("max_drawdown"),
                    "average_turnover": metrics.get("average_turnover"),
                }
            )
        return pd.DataFrame(rows)

    def find(
        self,
        strategy: str | None = None,
        version: str | None = None,
    ) -> list[StrategyRunRecord]:
        records = self.list_runs()
        if strategy is not None:
            records = [
                record
                for record in records
                if record.manifest.get("strategy_name") == strategy
            ]
        if version is not None:
            records = [
                record
                for record in records
                if record.manifest.get("strategy_version") == version
            ]
        return records
