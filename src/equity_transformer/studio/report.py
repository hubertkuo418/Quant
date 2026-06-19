from __future__ import annotations

from pathlib import Path

import pandas as pd

from equity_transformer.studio.registry import StrategyRunRegistry


def build_studio_report(
    output_path: str | Path,
    runs_root: str | Path = "artifacts/studio/runs",
    core_comparison_path: str | Path = (
        "artifacts/studio/comparisons/core_strategies/comparison.csv"
    ),
    optimization_path: str | Path = (
        "artifacts/studio/optimizations/factor_search/results.csv"
    ),
    recommendations_path: str | Path = (
        "artifacts/studio/recommendations/recommendations.csv"
    ),
) -> str:
    registry = StrategyRunRegistry(runs_root)
    runs = registry.summary()
    comparison = pd.read_csv(core_comparison_path)
    optimization = pd.read_csv(optimization_path)
    recommendations = pd.read_csv(recommendations_path)
    lines = [
        "# QuantLab Strategy Studio Results",
        "",
        "> Generated from registered strategy artifacts. Historical results are not "
        "investment advice or future-return guarantees.",
        "",
        "## Platform Snapshot",
        "",
        f"- Registered strategy runs: {len(runs)}",
        f"- Distinct strategy names: {runs['strategy'].nunique()}",
        f"- Optimizer candidates: {len(optimization)}",
        f"- Feasible candidates: {int(optimization['feasible'].map(_as_bool).sum())}",
        f"- Pareto candidates: "
        f"{int(optimization['pareto_efficient'].map(_as_bool).sum())}",
        "",
        "## Core Strategy Comparison",
        "",
        _markdown_table(
            comparison,
            [
                "portfolio",
                "common_start",
                "common_end",
                "annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "average_turnover",
            ],
        ),
        "",
        "## Profile Recommendations",
        "",
        _markdown_table(
            recommendations,
            [
                "recommendation_rank",
                "run_id",
                "recommendation_score",
                "annual_return",
                "sharpe_ratio",
                "max_drawdown",
                "average_turnover",
                "rationale",
            ],
        ),
        "",
        "## Interpretation Boundaries",
        "",
        "- The live data window is short and uses a present-day static universe.",
        "- Nasdaq close is used as adjusted close; corporate actions may distort "
        "returns.",
        "- Optimizer results are recomputed on a common period before ranking.",
        "- Execution lag 1/2 sensitivity is included in recommendation robustness.",
        "- Candidate rankings describe historical fit under configured constraints.",
        "",
    ]
    report = "\n".join(lines)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    return report


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, columns].copy()
    for column in selected.select_dtypes(include="number"):
        selected[column] = selected[column].map(lambda value: f"{value:.4f}")
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [
        "| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |"
        for row in selected.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}
