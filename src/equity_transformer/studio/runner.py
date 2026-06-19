from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd

from equity_transformer.backtest.config import BacktestConfig
from equity_transformer.backtest.engine import BacktestEngine
from equity_transformer.strategies.config import StrategyConfig
from equity_transformer.strategies.pipeline import StrategyPipeline
from equity_transformer.studio.specs import StrategySpec, save_strategy_spec


@dataclass(frozen=True)
class StrategyRunResult:
    run_id: str
    run_dir: Path
    metrics: dict[str, float]


class StrategyStudioRunner:
    def __init__(self, runs_root: str | Path = "artifacts/studio/runs") -> None:
        self.runs_root = Path(runs_root)

    def run(self, spec: StrategySpec) -> StrategyRunResult:
        spec.validate()
        self._validate_inputs(spec)
        created = datetime.now(UTC)
        run_id = f"{created:%Y%m%dT%H%M%S%fZ}_{spec.slug}_{spec.spec_hash[:8]}"
        run_dir = self.runs_root / run_id
        if run_dir.exists():
            raise FileExistsError(f"Strategy run already exists: {run_id}")
        run_dir.mkdir(parents=True)

        weights_path = run_dir / "target_weights.parquet"
        strategy_config = StrategyConfig(
            signal_path=spec.signal.path,
            output_path=weights_path,
            metadata_path=run_dir / "strategy_manifest.json",
            date_column=spec.signal.date_column,
            ticker_column=spec.signal.ticker_column,
            score_column=spec.signal.score_column,
            strategy_type=spec.portfolio.strategy_type,
            top_k=spec.portfolio.top_k,
            long_quantile=spec.portfolio.long_quantile,
            short_quantile=spec.portfolio.short_quantile,
            weighting=spec.portfolio.weighting,
            rebalance_frequency=spec.portfolio.rebalance_frequency,
            volatility_column=spec.signal.volatility_column,
            sector_column=spec.signal.sector_column,
            max_sector_weight=spec.risk.max_sector_weight,
            max_position_weight=spec.risk.max_position_weight,
            excluded_tickers=spec.universe.excluded_tickers,
            start_date=spec.universe.start_date,
            end_date=spec.universe.end_date,
        )
        signals = self._prepare_signals(spec)
        weights = StrategyPipeline(strategy_config).run(signals)
        backtest_config = BacktestConfig(
            market_path=spec.market_path,
            weights_path=weights_path,
            output_dir=run_dir / "backtest",
            initial_capital=spec.execution.initial_capital,
            commission_bps=spec.execution.commission_bps,
            slippage_bps=spec.execution.slippage_bps,
            annualization_factor=spec.annualization_factor,
            risk_free_rate=spec.risk_free_rate,
            execution_lag_days=spec.execution.execution_lag_days,
            min_dollar_volume=spec.risk.min_dollar_volume,
            liquidity_window=spec.risk.liquidity_window,
            benchmark_ticker=spec.benchmark_ticker,
            annual_cash_rate=spec.execution.annual_cash_rate,
            annual_borrow_rate=spec.execution.annual_borrow_rate,
        )
        equity, _, metrics = BacktestEngine(backtest_config).run(
            target_weights=weights
        )
        save_strategy_spec(spec, run_dir / "strategy.yaml")
        manifest = self._manifest(spec, run_id, created, weights, equity, metrics)
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return StrategyRunResult(run_id=run_id, run_dir=run_dir, metrics=metrics)

    @staticmethod
    def _prepare_signals(spec: StrategySpec) -> pd.DataFrame:
        signals = pd.read_parquet(spec.signal.path)
        if not spec.signal.components:
            return signals
        required = {
            spec.signal.date_column,
            spec.signal.ticker_column,
            *(component.column for component in spec.signal.components),
        }
        missing = required.difference(signals.columns)
        if missing:
            raise ValueError(f"Signal components missing columns: {sorted(missing)}")
        score = pd.Series(0.0, index=signals.index)
        dates = signals[spec.signal.date_column]
        for component in spec.signal.components:
            values = pd.to_numeric(signals[component.column], errors="coerce")
            transformed = _transform_component(values, dates, component.transform)
            score = score + component.weight * transformed
        signals[spec.signal.score_column] = score
        return signals

    @staticmethod
    def _validate_inputs(spec: StrategySpec) -> None:
        for label, path in {
            "market": spec.market_path,
            "signal": spec.signal.path,
        }.items():
            if not path.exists():
                raise FileNotFoundError(f"Strategy {label} file does not exist: {path}")

    def _manifest(
        self,
        spec: StrategySpec,
        run_id: str,
        created: datetime,
        weights: pd.DataFrame,
        equity: pd.DataFrame,
        metrics: dict[str, float],
    ) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "created_utc": created.isoformat(),
            "strategy_name": spec.name,
            "strategy_version": spec.version,
            "parent_run_id": spec.parent_run_id,
            "spec_hash": spec.spec_hash,
            "market_sha256": _sha256(spec.market_path),
            "signal_sha256": _sha256(spec.signal.path),
            "git_commit": _git_commit(),
            "packages": _package_versions(),
            "weights_rows": len(weights),
            "rebalance_dates": int(weights["date"].nunique()),
            "backtest_start": pd.to_datetime(equity["date"]).min().isoformat(),
            "backtest_end": pd.to_datetime(equity["date"]).max().isoformat(),
            "observations": len(equity),
            "metrics": metrics,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _package_versions() -> dict[str, str]:
    packages = {}
    for name in ("numpy", "pandas", "pyarrow", "scikit-learn", "torch"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            continue
    return packages


def _transform_component(
    values: pd.Series,
    dates: pd.Series,
    transform: str,
) -> pd.Series:
    if transform == "raw":
        return values
    grouped = values.groupby(pd.to_datetime(dates))
    if transform == "cross_sectional_rank":
        return grouped.rank(pct=True) - 0.5
    if transform == "cross_sectional_zscore":
        mean = grouped.transform("mean")
        scale = grouped.transform(lambda group: group.std(ddof=0))
        return ((values - mean) / scale.mask(scale.eq(0))).fillna(0.0)
    raise ValueError(f"Unsupported signal transform: {transform}")
