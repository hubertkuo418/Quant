from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from equity_transformer.studio.report import build_studio_report


def test_build_studio_report_from_registered_artifacts(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    run = runs / "run-1"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "strategy_name": "momentum",
                "strategy_version": "1.0.0",
                "metrics": {"sharpe_ratio": 1.0},
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.csv"
    pd.DataFrame(
        {
            "portfolio": ["run-1"],
            "common_start": ["2024-01-01"],
            "common_end": ["2024-02-01"],
            "annual_return": [0.2],
            "sharpe_ratio": [1.0],
            "max_drawdown": [-0.1],
            "average_turnover": [0.2],
        }
    ).to_csv(comparison, index=False)
    optimization = tmp_path / "optimization.csv"
    pd.DataFrame(
        {"feasible": [True, False], "pareto_efficient": [True, False]}
    ).to_csv(optimization, index=False)
    recommendations = tmp_path / "recommendations.csv"
    pd.DataFrame(
        {
            "recommendation_rank": [1],
            "run_id": ["run-1"],
            "recommendation_score": [0.8],
            "annual_return": [0.2],
            "sharpe_ratio": [1.0],
            "max_drawdown": [-0.1],
            "average_turnover": [0.2],
            "rationale": ["balanced profile"],
        }
    ).to_csv(recommendations, index=False)
    walk_forward_folds = tmp_path / "folds.csv"
    pd.DataFrame(
        {
            "fold": [1],
            "train_start": ["2024-01-01"],
            "train_end": ["2024-03-01"],
            "test_start": ["2024-03-08"],
            "test_end": ["2024-04-08"],
            "total_return": [0.05],
            "sharpe_ratio": [1.2],
            "max_drawdown": [-0.04],
        }
    ).to_csv(walk_forward_folds, index=False)
    walk_forward_metrics = tmp_path / "walk_forward_metrics.json"
    walk_forward_metrics.write_text(
        json.dumps(
            {
                "total_return": 0.05,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.04,
            }
        ),
        encoding="utf-8",
    )
    robustness_summary = tmp_path / "robustness.csv"
    pd.DataFrame(
        {
            "scenario": ["double_costs"],
            "oos_observations": [20],
            "sharpe_ratio": [0.8],
            "max_drawdown": [-0.05],
            "fold_sharpe_min": [-0.1],
            "positive_fold_rate": [0.5],
            "passes_constraints": [True],
        }
    ).to_csv(robustness_summary, index=False)
    robustness_aggregate = tmp_path / "robustness.json"
    robustness_aggregate.write_text(
        json.dumps(
            {
                "scenario_count": 1,
                "pass_rate": 1.0,
                "worst_sharpe": 0.8,
                "worst_max_drawdown": -0.05,
            }
        ),
        encoding="utf-8",
    )

    report = build_studio_report(
        tmp_path / "report.md",
        runs_root=runs,
        core_comparison_path=comparison,
        optimization_path=optimization,
        recommendations_path=recommendations,
        walk_forward_folds_path=walk_forward_folds,
        walk_forward_metrics_path=walk_forward_metrics,
        robustness_summary_path=robustness_summary,
        robustness_aggregate_path=robustness_aggregate,
    )

    assert "Registered strategy runs: 1" in report
    assert "| run-1 |" in report
    assert "balanced profile" in report
    assert "Frozen-Strategy Walk-Forward OOS" in report
    assert "Folds: 1" in report
    assert "Unified OOS Robustness" in report
