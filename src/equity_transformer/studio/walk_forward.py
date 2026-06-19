from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from equity_transformer.backtest.metrics import performance_metrics
from equity_transformer.studio.runner import StrategyStudioRunner
from equity_transformer.studio.specs import (
    StrategySpec,
    load_strategy_spec,
    save_strategy_spec,
)


@dataclass(frozen=True)
class WalkForwardConfig:
    strategy_spec_path: Path
    output_dir: Path
    train_days: int = 120
    test_days: int = 20
    step_days: int = 20
    purge_days: int = 1
    anchored_train: bool = False
    start_date: str | None = None
    end_date: str | None = None
    first_test_date: str | None = None

    def validate(self) -> None:
        if min(self.train_days, self.test_days, self.step_days) <= 0:
            raise ValueError("train_days, test_days, and step_days must be positive.")
        if self.purge_days < 0:
            raise ValueError("purge_days cannot be negative.")
        if self.step_days < self.test_days:
            raise ValueError("step_days must be at least test_days to avoid overlap.")


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


@dataclass(frozen=True)
class WalkForwardResult:
    output_dir: Path
    folds: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: dict[str, float]


def load_walk_forward_config(path: str | Path) -> WalkForwardConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    config = WalkForwardConfig(
        strategy_spec_path=Path(payload["strategy_spec_path"]),
        output_dir=Path(payload["output_dir"]),
        train_days=int(payload.get("train_days", 120)),
        test_days=int(payload.get("test_days", 20)),
        step_days=int(payload.get("step_days", payload.get("test_days", 20))),
        purge_days=int(payload.get("purge_days", 1)),
        anchored_train=bool(payload.get("anchored_train", False)),
        start_date=_optional_string(payload.get("start_date")),
        end_date=_optional_string(payload.get("end_date")),
        first_test_date=_optional_string(payload.get("first_test_date")),
    )
    config.validate()
    return config


class WalkForwardEvaluator:
    def __init__(self, config: WalkForwardConfig) -> None:
        config.validate()
        self.config = config

    def run(self) -> WalkForwardResult:
        spec = load_strategy_spec(self.config.strategy_spec_path)
        dates = self._available_dates(spec)
        folds = build_walk_forward_folds(dates, self.config)
        first_test_date = self.config.first_test_date or spec.universe.start_date
        if first_test_date:
            boundary = pd.Timestamp(first_test_date)
            folds = [fold for fold in folds if fold.test_start >= boundary]
        if not folds:
            raise ValueError("No complete walk-forward folds fit the available dates.")

        output = self.config.output_dir
        output.mkdir(parents=True, exist_ok=True)
        runner = StrategyStudioRunner(output / "runs")
        fold_rows = []
        oos_curves = []
        for fold in folds:
            fold_spec = replace(
                spec,
                version=f"{spec.version}-wf{fold.fold:02d}",
                universe=replace(
                    spec.universe,
                    start_date=fold.test_start.strftime("%Y-%m-%d"),
                    end_date=fold.test_end.strftime("%Y-%m-%d"),
                ),
            )
            result = runner.run(fold_spec)
            curve = pd.read_csv(result.run_dir / "backtest" / "equity_curve.csv")
            curve["date"] = pd.to_datetime(curve["date"])
            curve = curve.loc[
                curve["date"].between(fold.test_start, fold.test_end)
            ].copy()
            if curve.empty:
                raise ValueError(f"Fold {fold.fold} produced no OOS equity rows.")
            in_fold = dates.to_series().between(fold.test_start, fold.test_end)
            fold_dates = dates[in_fold]
            curve = self._align_fold_calendar(curve, fold_dates, fold.fold)
            oos_curves.append(curve)
            fold_equity = self._stitch([curve], spec.execution.initial_capital)
            fold_metrics = performance_metrics(
                fold_equity,
                spec.annualization_factor,
                spec.risk_free_rate,
            )
            fold_rows.append(
                {
                    "fold": fold.fold,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "run_id": result.run_id,
                    **fold_metrics,
                }
            )

        stitched = self._stitch(oos_curves, spec.execution.initial_capital)
        metrics = performance_metrics(
            stitched,
            spec.annualization_factor,
            spec.risk_free_rate,
        )
        fold_table = pd.DataFrame(fold_rows)
        fold_table.to_csv(output / "folds.csv", index=False)
        stitched.to_csv(output / "oos_equity_curve.csv", index=False)
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=True),
            encoding="utf-8",
        )
        save_strategy_spec(spec, output / "strategy.yaml")
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "evaluation_mode": "frozen_strategy_rolling_oos",
            "first_test_date": first_test_date,
            "strategy_spec_hash": spec.spec_hash,
            "config": _stringify(asdict(self.config)),
            "fold_count": len(folds),
            "oos_start": stitched["date"].min().isoformat(),
            "oos_end": stitched["date"].max().isoformat(),
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return WalkForwardResult(output, fold_table, stitched, metrics)

    def _available_dates(self, spec: StrategySpec) -> pd.DatetimeIndex:
        market = pd.read_parquet(spec.market_path, columns=["date"])
        signals = pd.read_parquet(
            spec.signal.path,
            columns=[spec.signal.date_column],
        )
        market_dates = pd.DatetimeIndex(pd.to_datetime(market["date"]).unique())
        signal_dates = pd.DatetimeIndex(
            pd.to_datetime(signals[spec.signal.date_column]).unique()
        )
        dates = market_dates.intersection(signal_dates).sort_values()
        if self.config.start_date:
            dates = dates[dates >= pd.Timestamp(self.config.start_date)]
        if self.config.end_date:
            dates = dates[dates <= pd.Timestamp(self.config.end_date)]
        return dates

    @staticmethod
    def _stitch(curves: list[pd.DataFrame], initial_capital: float) -> pd.DataFrame:
        stitched = (
            pd.concat(curves, ignore_index=True)
            .sort_values(["date", "fold"])
            .drop_duplicates("date", keep="first")
            .reset_index(drop=True)
        )
        stitched["nav"] = initial_capital * (1 + stitched["net_return"]).cumprod()
        initial = {column: 0.0 for column in stitched.columns}
        initial["date"] = stitched["date"].min() - pd.offsets.BDay(1)
        initial["nav"] = initial_capital
        initial["fold"] = 0
        return pd.concat([pd.DataFrame([initial]), stitched], ignore_index=True)

    @staticmethod
    def _align_fold_calendar(
        curve: pd.DataFrame,
        dates: pd.DatetimeIndex,
        fold: int,
    ) -> pd.DataFrame:
        aligned = curve.set_index("date").reindex(dates).rename_axis("date")
        numeric_columns = aligned.select_dtypes(include="number").columns
        aligned[numeric_columns] = aligned[numeric_columns].fillna(0.0)
        aligned["fold"] = fold
        return aligned.reset_index()


def build_walk_forward_folds(
    dates: pd.DatetimeIndex,
    config: WalkForwardConfig,
) -> list[WalkForwardFold]:
    config.validate()
    ordered = pd.DatetimeIndex(dates).drop_duplicates().sort_values()
    test_start_index = config.train_days + config.purge_days
    folds = []
    fold_number = 1
    while test_start_index + config.test_days <= len(ordered):
        train_end_index = test_start_index - config.purge_days - 1
        train_start_index = (
            0
            if config.anchored_train
            else train_end_index - config.train_days + 1
        )
        test_end_index = test_start_index + config.test_days - 1
        folds.append(
            WalkForwardFold(
                fold=fold_number,
                train_start=ordered[train_start_index],
                train_end=ordered[train_end_index],
                test_start=ordered[test_start_index],
                test_end=ordered[test_end_index],
            )
        )
        fold_number += 1
        test_start_index += config.step_days
    return folds


def _optional_string(value: object) -> str | None:
    return str(value) if value is not None and str(value).strip() else None


def _stringify(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _stringify(item) for key, item in value.items()}
    return value
