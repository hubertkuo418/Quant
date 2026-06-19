from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ReportConfig:
    baselines_metrics_path: Path
    recurrent_metrics_path: Path
    transformer_metrics_path: Path
    backtest_metrics_path: Path
    trade_log_path: Path
    exposure_path: Path
    sector_exposure_path: Path | None
    benchmark_comparison_path: Path
    output_dir: Path


def load_report_config(path: str | Path) -> ReportConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return ReportConfig(
        baselines_metrics_path=Path(payload["baselines_metrics_path"]),
        recurrent_metrics_path=Path(payload["recurrent_metrics_path"]),
        transformer_metrics_path=Path(payload["transformer_metrics_path"]),
        backtest_metrics_path=Path(payload["backtest_metrics_path"]),
        trade_log_path=Path(payload["trade_log_path"]),
        exposure_path=Path(payload["exposure_path"]),
        sector_exposure_path=(
            Path(payload["sector_exposure_path"])
            if payload.get("sector_exposure_path") is not None
            else None
        ),
        benchmark_comparison_path=Path(payload["benchmark_comparison_path"]),
        output_dir=Path(payload["output_dir"]),
    )
