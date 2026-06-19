from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from equity_transformer.reporting.equity_comparison import compare_equity_curves
from equity_transformer.studio.registry import StrategyRunRegistry


def compare_strategy_runs(
    run_ids: list[str] | tuple[str, ...],
    registry: StrategyRunRegistry,
    output_dir: str | Path,
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    unique_ids = list(dict.fromkeys(run_ids))
    if len(unique_ids) < 2:
        raise ValueError("At least two unique strategy runs are required.")
    records = [registry.get(run_id) for run_id in unique_ids]
    curves = {
        record.run_id: pd.read_csv(record.path / "backtest" / "equity_curve.csv")
        for record in records
    }
    comparison, aligned = compare_equity_curves(
        curves,
        annualization_factor=annualization_factor,
        risk_free_rate=risk_free_rate,
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "comparison.csv", index=False)
    aligned_dir = output / "aligned_curves"
    aligned_dir.mkdir(exist_ok=True)
    for run_id, frame in aligned.items():
        frame.to_csv(aligned_dir / f"{run_id}.csv", index=False)
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "run_ids": unique_ids,
        "common_start": str(comparison["common_start"].iloc[0]),
        "common_end": str(comparison["common_end"].iloc[0]),
        "annualization_factor": annualization_factor,
        "risk_free_rate": risk_free_rate,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return comparison
