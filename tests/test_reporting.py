from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from equity_transformer.reporting.config import ReportConfig
from equity_transformer.reporting.summary import (
    ReportPipeline,
    build_model_comparison,
    build_portfolio_comparison,
    build_trading_diagnostics,
)


def test_build_model_comparison_reads_baseline_and_transformer(tmp_path: Path) -> None:
    baseline_path = tmp_path / "metrics.csv"
    transformer_path = tmp_path / "metrics.json"
    pd.DataFrame(
        {
            "model": ["ridge"],
            "horizon": [5],
            "rmse": [0.2],
            "mae": [0.1],
        }
    ).to_csv(baseline_path, index=False)
    transformer_path.write_text(
        json.dumps({"target_horizon": 5, "metrics": {"rmse": 0.15, "mae": 0.08}}),
        encoding="utf-8",
    )

    recurrent_path = tmp_path / "recurrent.csv"
    pd.DataFrame({"model": ["gru"], "horizon": [5], "rmse": [0.18]}).to_csv(
        recurrent_path,
        index=False,
    )

    comparison = build_model_comparison(baseline_path, recurrent_path, transformer_path)

    assert comparison["model"].tolist() == ["transformer_v1", "gru", "ridge"]
    assert set(comparison["source"]) == {"baseline", "recurrent", "transformer"}


def test_build_model_comparison_reads_transformer_metrics_by_horizon(
    tmp_path: Path,
) -> None:
    transformer_path = tmp_path / "metrics.json"
    transformer_path.write_text(
        json.dumps(
            {
                "target_horizon": None,
                "metrics": {"rmse": 0.3},
                "metrics_by_horizon": {
                    "5": {"rmse": 0.2, "mae": 0.1},
                    "20": {"rmse": 0.4, "mae": 0.2},
                },
            }
        ),
        encoding="utf-8",
    )

    comparison = build_model_comparison(
        tmp_path / "missing_baseline.csv",
        tmp_path / "missing_recurrent.csv",
        transformer_path,
    )

    assert comparison["horizon"].tolist() == [5, 20]
    assert comparison["rmse"].tolist() == [0.2, 0.4]
    assert set(comparison["source"]) == {"transformer"}


def test_empty_model_comparison_retains_output_schema(tmp_path: Path) -> None:
    comparison = build_model_comparison(
        tmp_path / "missing_baseline.csv",
        tmp_path / "missing_recurrent.csv",
        tmp_path / "missing_transformer.json",
    )

    assert comparison.empty
    assert {"model", "horizon", "rmse", "source"}.issubset(comparison.columns)


def test_build_portfolio_comparison_reads_strategy_and_benchmarks(
    tmp_path: Path,
) -> None:
    strategy_path = tmp_path / "strategy.json"
    benchmark_path = tmp_path / "benchmarks.csv"
    strategy_path.write_text(
        json.dumps({"sharpe_ratio": 1.0, "total_return": 0.2}),
        encoding="utf-8",
    )
    pd.DataFrame(
        {"benchmark": ["equal_weight"], "sharpe_ratio": [0.5], "total_return": [0.1]}
    ).to_csv(benchmark_path, index=False)

    comparison = build_portfolio_comparison(strategy_path, benchmark_path)

    assert comparison["portfolio"].tolist() == ["strategy", "equal_weight"]


def test_report_pipeline_writes_outputs(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.csv"
    transformer = tmp_path / "transformer.json"
    backtest = tmp_path / "backtest.json"
    benchmark = tmp_path / "benchmark.csv"
    trade_log = tmp_path / "trade_log.parquet"
    exposure = tmp_path / "exposure.csv"
    sector_exposure = tmp_path / "sector_exposure.csv"
    pd.DataFrame({"model": ["ridge"], "horizon": [5], "rmse": [0.2]}).to_csv(
        baseline,
        index=False,
    )
    transformer.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
    backtest.write_text(json.dumps({"sharpe_ratio": 1.0}), encoding="utf-8")
    pd.DataFrame({"benchmark": ["ew"], "sharpe_ratio": [0.5]}).to_csv(
        benchmark,
        index=False,
    )
    pd.DataFrame({"abs_trade_value": [10.0], "cost": [0.1]}).to_parquet(
        trade_log,
        index=False,
    )
    pd.DataFrame(
        {
            "gross_exposure": [1.0],
            "net_exposure": [1.0],
            "active_positions": [2],
        }
    ).to_csv(exposure, index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "sector": ["Technology", "Utilities"],
            "gross_exposure": [0.7, 0.3],
        }
    ).to_csv(sector_exposure, index=False)
    config = ReportConfig(
        baselines_metrics_path=baseline,
        recurrent_metrics_path=tmp_path / "missing_recurrent.csv",
        transformer_metrics_path=transformer,
        backtest_metrics_path=backtest,
        trade_log_path=trade_log,
        exposure_path=exposure,
        sector_exposure_path=sector_exposure,
        benchmark_comparison_path=benchmark,
        output_dir=tmp_path / "reports",
    )

    outputs = ReportPipeline(config).run()

    assert not outputs["model_comparison"].empty
    assert not outputs["portfolio_comparison"].empty
    assert not outputs["trading_diagnostics"].empty
    assert outputs["trading_diagnostics"].loc[0, "sector_count"] == 2
    assert (config.output_dir / "model_comparison.csv").exists()


def test_build_trading_diagnostics_summarizes_trades_and_exposure(
    tmp_path: Path,
) -> None:
    trade_log = tmp_path / "trade_log.parquet"
    exposure = tmp_path / "exposure.csv"
    sector_exposure = tmp_path / "sector_exposure.csv"
    pd.DataFrame(
        {
            "abs_trade_value": [100.0, 50.0],
            "cost": [1.0, 0.5],
        }
    ).to_parquet(trade_log, index=False)
    pd.DataFrame(
        {
            "gross_exposure": [1.0, 0.8],
            "net_exposure": [1.0, 0.8],
            "active_positions": [2, 3],
        }
    ).to_csv(exposure, index=False)
    pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "sector": ["Technology", "Utilities", "Technology"],
            "gross_exposure": [0.6, 0.4, 0.8],
        }
    ).to_csv(sector_exposure, index=False)

    diagnostics = build_trading_diagnostics(trade_log, exposure, sector_exposure)

    assert diagnostics.loc[0, "trade_count"] == 2
    assert diagnostics.loc[0, "total_trade_cost"] == 1.5
    assert diagnostics.loc[0, "average_active_positions"] == 2.5
    assert diagnostics.loc[0, "max_sector_gross_exposure"] == 0.8
    assert diagnostics.loc[0, "average_largest_sector_gross_exposure"] == 0.7
