from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_transformer.backtest.metrics import performance_metrics


@dataclass(frozen=True)
class EquityCurveSource:
    name: str
    path: Path


@dataclass(frozen=True)
class EquityComparisonConfig:
    sources: tuple[EquityCurveSource, ...]
    output_path: Path
    metadata_path: Path
    annualization_factor: int
    risk_free_rate: float


def load_equity_comparison_config(path: str | Path) -> EquityComparisonConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return EquityComparisonConfig(
        sources=tuple(
            EquityCurveSource(str(item["name"]), Path(item["path"]))
            for item in payload["sources"]
        ),
        output_path=Path(payload["output_path"]),
        metadata_path=Path(payload["metadata_path"]),
        annualization_factor=int(payload.get("annualization_factor", 252)),
        risk_free_rate=float(payload.get("risk_free_rate", 0.0)),
    )


def compare_equity_curves(
    curves: dict[str, pd.DataFrame],
    annualization_factor: int = 252,
    risk_free_rate: float = 0.0,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if len(curves) < 2:
        raise ValueError("At least two equity curves are required for comparison.")
    prepared = {name: _prepare_curve(frame) for name, frame in curves.items()}
    common_start = max(frame["date"].min() for frame in prepared.values())
    common_end = min(frame["date"].max() for frame in prepared.values())
    if common_start >= common_end:
        raise ValueError("Equity curves do not have a usable overlapping period.")

    aligned = {}
    rows = []
    for name, frame in prepared.items():
        common = frame.loc[
            frame["date"].between(common_start, common_end)
        ].copy()
        common = _rebase_common_period(common)
        aligned[name] = common
        metrics = performance_metrics(
            common,
            annualization_factor=annualization_factor,
            risk_free_rate=risk_free_rate,
        )
        rows.append(
            {
                "portfolio": name,
                "common_start": common_start,
                "common_end": common_end,
                "observations": len(common) - 1,
                **metrics,
            }
        )
    comparison = pd.DataFrame(rows).sort_values(
        "sharpe_ratio", ascending=False, na_position="last"
    )
    return comparison.reset_index(drop=True), aligned


class EquityComparisonPipeline:
    def __init__(self, config: EquityComparisonConfig) -> None:
        self.config = config

    def run(
        self,
        curves: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        self._validate_sources()
        frames = curves or {
            source.name: pd.read_csv(source.path) for source in self.config.sources
        }
        comparison, aligned = compare_equity_curves(
            frames,
            self.config.annualization_factor,
            self.config.risk_free_rate,
        )
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(self.config.output_path, index=False)
        aligned_dir = self.config.output_path.parent / "aligned_curves"
        aligned_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in aligned.items():
            frame.to_csv(aligned_dir / f"{name}.csv", index=False)
        self.config.metadata_path.write_text(
            json.dumps(
                {
                    "sources": [
                        {"name": source.name, "path": str(source.path)}
                        for source in self.config.sources
                    ],
                    "common_start": str(comparison["common_start"].iloc[0]),
                    "common_end": str(comparison["common_end"].iloc[0]),
                    "portfolios": comparison["portfolio"].tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return comparison

    def _validate_sources(self) -> None:
        names = [source.name for source in self.config.sources]
        if len(names) < 2:
            raise ValueError("At least two equity curve sources are required.")
        if len(names) != len(set(names)):
            raise ValueError("Equity curve source names must be unique.")
        for name in names:
            if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
                raise ValueError(f"Invalid equity curve source name: {name}")


def _prepare_curve(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "net_return"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Equity curve missing columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result = result.sort_values("date").drop_duplicates("date", keep="last")
    if "turnover" not in result.columns:
        result["turnover"] = 0.0
    if "cost" not in result.columns:
        result["cost"] = 0.0
    return result


def _rebase_common_period(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame) < 2:
        raise ValueError("Common equity period must contain at least two rows.")
    result = frame.reset_index(drop=True).copy()
    result.loc[0, ["net_return", "turnover", "cost"]] = 0.0
    result["nav"] = (1 + result["net_return"].fillna(0.0)).cumprod()
    return result
