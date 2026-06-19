from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def read_csv_if_exists(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def read_parquet_if_exists(path: str | Path) -> pd.DataFrame:
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return pd.read_parquet(file_path)


def read_json_if_exists(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        return {}
    return json.loads(file_path.read_text(encoding="utf-8"))


def load_dashboard_artifacts(root: str | Path = ".") -> dict[str, object]:
    base = Path(root)
    return {
        "data_quality_summary": read_json_if_exists(
            base / "artifacts/data_quality/summary.json"
        ),
        "data_quality_issues": read_csv_if_exists(
            base / "artifacts/data_quality/issues.csv"
        ),
        "factor_ic": read_csv_if_exists(base / "artifacts/factors/ic_summary.csv"),
        "factor_quantiles": read_csv_if_exists(
            base / "artifacts/factors/quantile_summary.csv"
        ),
        "selected_factors": read_csv_if_exists(
            base / "artifacts/factors/selected_factors.csv"
        ),
        "selected_factors_manifest": read_json_if_exists(
            base / "artifacts/factors/selected_factors.json"
        ),
        "factor_signals": read_parquet_if_exists(
            base / "artifacts/factors/factor_signals.parquet"
        ),
        "factor_signals_manifest": read_json_if_exists(
            base / "artifacts/factors/factor_signals.json"
        ),
        "strategy_weights": read_parquet_if_exists(
            base / "artifacts/strategies/target_weights.parquet"
        ),
        "strategy_manifest": read_json_if_exists(
            base / "artifacts/strategies/manifest.json"
        ),
        "model_signals": read_parquet_if_exists(
            base / "artifacts/strategies/model_signals.parquet"
        ),
        "backtest_equity": read_csv_if_exists(
            base / "artifacts/backtests/equity_curve.csv"
        ),
        "backtest_exposure": read_csv_if_exists(
            base / "artifacts/backtests/exposure.csv"
        ),
        "backtest_sector_exposure": read_csv_if_exists(
            base / "artifacts/backtests/sector_exposure.csv"
        ),
        "backtest_trades": read_parquet_if_exists(
            base / "artifacts/backtests/trade_log.parquet"
        ),
        "backtest_metrics": read_json_if_exists(
            base / "artifacts/backtests/metrics.json"
        ),
        "benchmark_comparison": read_csv_if_exists(
            base / "artifacts/benchmarks/comparison.csv"
        ),
        "regime_performance": read_csv_if_exists(
            base / "artifacts/regimes/regime_performance.csv"
        ),
        "daily_regimes": read_csv_if_exists(
            base / "artifacts/regimes/daily_regimes.csv"
        ),
        "regime_manifest": read_json_if_exists(
            base / "artifacts/regimes/manifest.json"
        ),
        "sensitivity_comparison": read_csv_if_exists(
            base / "artifacts/sensitivity/comparison.csv"
        ),
        "sensitivity_manifest": read_json_if_exists(
            base / "artifacts/sensitivity/manifest.json"
        ),
        "attribution_summary": read_csv_if_exists(
            base / "artifacts/attribution/ticker_summary.csv"
        ),
        "attribution_metrics": read_json_if_exists(
            base / "artifacts/attribution/metrics.json"
        ),
        "common_period_comparison": read_csv_if_exists(
            base / "artifacts/comparisons/common_period.csv"
        ),
        "common_period_manifest": read_json_if_exists(
            base / "artifacts/comparisons/manifest.json"
        ),
        "model_comparison": read_csv_if_exists(
            base / "artifacts/reports/model_comparison.csv"
        ),
        "transformer_predictions": read_parquet_if_exists(
            base / "artifacts/transformer_v1/test_predictions.parquet"
        ),
        "transformer_metrics": read_json_if_exists(
            base / "artifacts/transformer_v1/metrics.json"
        ),
        "portfolio_comparison": read_csv_if_exists(
            base / "artifacts/reports/portfolio_comparison.csv"
        ),
        "trading_diagnostics": read_csv_if_exists(
            base / "artifacts/reports/trading_diagnostics.csv"
        ),
    }
