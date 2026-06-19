from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from equity_transformer.gui.artifacts import (
    load_dashboard_artifacts,
    read_csv_if_exists,
    read_json_if_exists,
    read_parquet_if_exists,
)


def test_optional_artifact_readers_return_empty_objects(tmp_path: Path) -> None:
    assert read_csv_if_exists(tmp_path / "missing.csv").empty
    assert read_parquet_if_exists(tmp_path / "missing.parquet").empty
    assert read_json_if_exists(tmp_path / "missing.json") == {}


def test_csv_reader_handles_existing_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_text("", encoding="utf-8")

    assert read_csv_if_exists(path).empty


def test_dashboard_artifact_loader_reads_existing_files(tmp_path: Path) -> None:
    factor_dir = tmp_path / "artifacts" / "factors"
    backtest_dir = tmp_path / "artifacts" / "backtests"
    benchmark_dir = tmp_path / "artifacts" / "benchmarks"
    strategy_dir = tmp_path / "artifacts" / "strategies"
    transformer_dir = tmp_path / "artifacts" / "transformer_v1"
    report_dir = tmp_path / "artifacts" / "reports"
    regime_dir = tmp_path / "artifacts" / "regimes"
    sensitivity_dir = tmp_path / "artifacts" / "sensitivity"
    attribution_dir = tmp_path / "artifacts" / "attribution"
    comparison_dir = tmp_path / "artifacts" / "comparisons"
    quality_dir = tmp_path / "artifacts" / "data_quality"
    for directory in [
        factor_dir,
        backtest_dir,
        benchmark_dir,
        strategy_dir,
        transformer_dir,
        report_dir,
        regime_dir,
        sensitivity_dir,
        attribution_dir,
        comparison_dir,
        quality_dir,
    ]:
        directory.mkdir(parents=True)

    pd.DataFrame({"factor": ["alpha"], "mean_rank_ic": [0.1]}).to_csv(
        factor_dir / "ic_summary.csv",
        index=False,
    )
    pd.DataFrame({"factor": ["alpha"], "selection_score": [0.2]}).to_csv(
        factor_dir / "selected_factors.csv",
        index=False,
    )
    (factor_dir / "selected_factors.json").write_text(
        json.dumps({"selected_factors": ["alpha"]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"date": ["2024-01-01"], "ticker": ["AAA"], "factor_score": [1.0]}
    ).to_parquet(factor_dir / "factor_signals.parquet", index=False)
    (factor_dir / "factor_signals.json").write_text(
        json.dumps({"score_column": "factor_score", "factor_count": 1}),
        encoding="utf-8",
    )
    pd.DataFrame({"date": ["2024-01-01"], "nav": [1.0]}).to_csv(
        backtest_dir / "equity_curve.csv",
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "gross_exposure": [1.0]}).to_csv(
        backtest_dir / "exposure.csv",
        index=False,
    )
    pd.DataFrame(
        {"date": ["2024-01-01"], "sector": ["Technology"], "gross_exposure": [0.6]}
    ).to_csv(
        backtest_dir / "sector_exposure.csv",
        index=False,
    )
    pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAA"]}).to_parquet(
        backtest_dir / "trade_log.parquet",
        index=False,
    )
    weights = pd.DataFrame(
        {"date": ["2024-01-01"], "ticker": ["AAA"], "weight": [1.0]}
    )
    weights.to_parquet(strategy_dir / "target_weights.parquet", index=False)
    pd.DataFrame(
        {"date": ["2024-01-01"], "ticker": ["AAA"], "model_score": [0.02]}
    ).to_parquet(strategy_dir / "model_signals.parquet", index=False)
    (strategy_dir / "manifest.json").write_text(
        json.dumps({"strategy_type": "long_only_top_k", "max_position_weight": 0.2}),
        encoding="utf-8",
    )
    pd.DataFrame({"model": ["transformer_v1"], "horizon": [5], "rmse": [0.1]}).to_csv(
        report_dir / "model_comparison.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "ticker": ["AAA"],
            "horizon": [5],
            "prediction": [0.02],
        }
    ).to_parquet(transformer_dir / "test_predictions.parquet", index=False)
    (transformer_dir / "metrics.json").write_text(
        json.dumps({"horizons": [5, 20], "metrics_by_horizon": {"5": {"rmse": 0.1}}}),
        encoding="utf-8",
    )
    (backtest_dir / "metrics.json").write_text(
        json.dumps({"sharpe_ratio": 1.2}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"regime": ["bull_low_vol"], "sharpe_ratio": [1.1]}
    ).to_csv(regime_dir / "regime_performance.csv", index=False)
    pd.DataFrame(
        {"date": ["2024-01-01"], "regime": ["bull_low_vol"]}
    ).to_csv(regime_dir / "daily_regimes.csv", index=False)
    (regime_dir / "manifest.json").write_text(
        json.dumps({"benchmark_ticker": "SPY"}), encoding="utf-8"
    )
    pd.DataFrame(
        {"scenario": ["double_costs"], "sharpe_ratio": [0.8]}
    ).to_csv(sensitivity_dir / "comparison.csv", index=False)
    (sensitivity_dir / "manifest.json").write_text(
        json.dumps({"scenarios": [{"name": "double_costs"}]}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"ticker": ["AAA"], "total_contribution": [0.1]}
    ).to_csv(attribution_dir / "ticker_summary.csv", index=False)
    (attribution_dir / "metrics.json").write_text(
        json.dumps({"effective_contributors": 1.0}), encoding="utf-8"
    )
    pd.DataFrame(
        {"portfolio": ["factor"], "sharpe_ratio": [1.0]}
    ).to_csv(comparison_dir / "common_period.csv", index=False)
    (comparison_dir / "manifest.json").write_text(
        json.dumps({"common_start": "2024-01-01"}), encoding="utf-8"
    )
    (quality_dir / "summary.json").write_text(
        json.dumps({"rows": 100, "tickers": 2, "issue_count": 1}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"ticker": ["AAA"], "metric": ["calendar_coverage"], "value": [0.8]}
    ).to_csv(quality_dir / "issues.csv", index=False)

    artifacts = load_dashboard_artifacts(tmp_path)

    assert not artifacts["factor_ic"].empty
    assert not artifacts["selected_factors"].empty
    assert artifacts["selected_factors_manifest"]["selected_factors"] == ["alpha"]
    assert not artifacts["factor_signals"].empty
    assert artifacts["factor_signals_manifest"]["factor_count"] == 1
    assert not artifacts["strategy_weights"].empty
    assert not artifacts["model_signals"].empty
    assert artifacts["strategy_manifest"]["max_position_weight"] == 0.2
    assert not artifacts["backtest_exposure"].empty
    assert not artifacts["backtest_sector_exposure"].empty
    assert not artifacts["backtest_trades"].empty
    assert artifacts["backtest_metrics"]["sharpe_ratio"] == 1.2
    assert artifacts["benchmark_comparison"].empty
    assert not artifacts["model_comparison"].empty
    assert not artifacts["transformer_predictions"].empty
    assert artifacts["transformer_metrics"]["horizons"] == [5, 20]
    assert not artifacts["regime_performance"].empty
    assert not artifacts["daily_regimes"].empty
    assert artifacts["regime_manifest"]["benchmark_ticker"] == "SPY"
    assert not artifacts["sensitivity_comparison"].empty
    assert artifacts["sensitivity_manifest"]["scenarios"][0]["name"] == (
        "double_costs"
    )
    assert not artifacts["attribution_summary"].empty
    assert artifacts["attribution_metrics"]["effective_contributors"] == 1.0
    assert not artifacts["common_period_comparison"].empty
    assert artifacts["common_period_manifest"]["common_start"] == "2024-01-01"
    assert artifacts["data_quality_summary"]["issue_count"] == 1
    assert not artifacts["data_quality_issues"].empty
