from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_transformer.data.validation import validate_market_frame


@dataclass(frozen=True)
class MarketQualityConfig:
    market_path: Path
    output_dir: Path
    min_calendar_coverage: float = 0.95
    max_zero_volume_rate: float = 0.01
    max_stale_price_run: int = 5
    max_abs_daily_return: float = 0.5


def load_market_quality_config(path: str | Path) -> MarketQualityConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    config = MarketQualityConfig(
        market_path=Path(payload["market_path"]),
        output_dir=Path(payload["output_dir"]),
        min_calendar_coverage=float(payload.get("min_calendar_coverage", 0.95)),
        max_zero_volume_rate=float(payload.get("max_zero_volume_rate", 0.01)),
        max_stale_price_run=int(payload.get("max_stale_price_run", 5)),
        max_abs_daily_return=float(payload.get("max_abs_daily_return", 0.5)),
    )
    _validate_config(config)
    return config


class MarketQualityAnalyzer:
    def __init__(self, config: MarketQualityConfig) -> None:
        _validate_config(config)
        self.config = config

    def run(self, market: pd.DataFrame | None = None) -> dict[str, Any]:
        frame = (
            pd.read_parquet(self.config.market_path)
            if market is None
            else market.copy()
        )
        frame["date"] = pd.to_datetime(frame["date"])
        validate_market_frame(frame)
        calendar = pd.Index(sorted(frame["date"].unique()))

        rows = [
            self._ticker_metrics(group, calendar)
            for _, group in frame.groupby("ticker")
        ]
        per_ticker = pd.DataFrame(rows).sort_values("ticker").reset_index(drop=True)
        issues = self._build_issues(per_ticker)
        summary = {
            "run_utc": datetime.now(UTC).isoformat(),
            "market_path": str(self.config.market_path),
            "rows": len(frame),
            "tickers": frame["ticker"].nunique(),
            "calendar_dates": len(calendar),
            "min_date": frame["date"].min().isoformat(),
            "max_date": frame["date"].max().isoformat(),
            "tickers_with_issues": int(issues["ticker"].nunique()),
            "issue_count": len(issues),
            "thresholds": {
                "min_calendar_coverage": self.config.min_calendar_coverage,
                "max_zero_volume_rate": self.config.max_zero_volume_rate,
                "max_stale_price_run": self.config.max_stale_price_run,
                "max_abs_daily_return": self.config.max_abs_daily_return,
            },
        }
        self._write_outputs(per_ticker, issues, summary)
        return {"per_ticker": per_ticker, "issues": issues, "summary": summary}

    @staticmethod
    def _ticker_metrics(group: pd.DataFrame, calendar: pd.Index) -> dict[str, Any]:
        ordered = group.sort_values("date")
        observed = pd.Index(ordered["date"].unique())
        missing = calendar.difference(observed)
        returns = ordered["adj_close"].pct_change(fill_method=None).abs()
        stale = ordered["adj_close"].eq(ordered["adj_close"].shift())
        return {
            "ticker": str(ordered["ticker"].iloc[0]),
            "rows": len(ordered),
            "first_date": ordered["date"].min(),
            "last_date": ordered["date"].max(),
            "calendar_coverage": len(observed) / len(calendar),
            "missing_dates": len(missing),
            "zero_volume_rate": float(ordered["volume"].eq(0).mean()),
            "longest_stale_price_run": _longest_true_run(stale),
            "max_abs_daily_return": float(returns.max())
            if returns.notna().any()
            else 0.0,
        }

    def _build_issues(self, metrics: pd.DataFrame) -> pd.DataFrame:
        rules = (
            (
                "calendar_coverage",
                "below_minimum",
                self.config.min_calendar_coverage,
                "lt",
            ),
            (
                "zero_volume_rate",
                "above_maximum",
                self.config.max_zero_volume_rate,
                "gt",
            ),
            (
                "longest_stale_price_run",
                "above_maximum",
                self.config.max_stale_price_run,
                "gt",
            ),
            (
                "max_abs_daily_return",
                "above_maximum",
                self.config.max_abs_daily_return,
                "gt",
            ),
        )
        issue_frames = []
        for metric, condition, threshold, operator in rules:
            mask = (
                metrics[metric] < threshold
                if operator == "lt"
                else metrics[metric] > threshold
            )
            flagged = metrics.loc[mask, ["ticker", metric]].rename(
                columns={metric: "value"}
            )
            flagged["metric"] = metric
            flagged["condition"] = condition
            flagged["threshold"] = threshold
            issue_frames.append(flagged)
        columns = ["ticker", "metric", "value", "condition", "threshold"]
        if not any(not frame.empty for frame in issue_frames):
            return pd.DataFrame(columns=columns)
        return (
            pd.concat(issue_frames, ignore_index=True)[columns]
            .sort_values(["ticker", "metric"])
            .reset_index(drop=True)
        )

    def _write_outputs(
        self,
        per_ticker: pd.DataFrame,
        issues: pd.DataFrame,
        summary: dict[str, Any],
    ) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        per_ticker.to_csv(self.config.output_dir / "per_ticker.csv", index=False)
        issues.to_csv(self.config.output_dir / "issues.csv", index=False)
        (self.config.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


def _longest_true_run(values: pd.Series) -> int:
    groups = values.ne(values.shift()).cumsum()
    runs = values.groupby(groups).sum()
    return int(runs.max()) if not runs.empty else 0


def _validate_config(config: MarketQualityConfig) -> None:
    for name in ("min_calendar_coverage", "max_zero_volume_rate"):
        value = getattr(config, name)
        if not 0 <= value <= 1:
            raise ValueError(f"{name} must be between 0 and 1.")
    if config.max_stale_price_run < 0:
        raise ValueError("max_stale_price_run cannot be negative.")
    if config.max_abs_daily_return <= 0:
        raise ValueError("max_abs_daily_return must be positive.")
