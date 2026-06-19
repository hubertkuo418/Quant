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

    report = build_studio_report(
        tmp_path / "report.md",
        runs_root=runs,
        core_comparison_path=comparison,
        optimization_path=optimization,
        recommendations_path=recommendations,
    )

    assert "Registered strategy runs: 1" in report
    assert "| run-1 |" in report
    assert "balanced profile" in report
