from __future__ import annotations

import json
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
    walk_forward_folds_path: str | Path = (
        "artifacts/studio/walk_forward/factor_top10/folds.csv"
    ),
    walk_forward_metrics_path: str | Path = (
        "artifacts/studio/walk_forward/factor_top10/metrics.json"
    ),
) -> str:
    registry = StrategyRunRegistry(runs_root)
    runs = registry.summary()
    comparison = pd.read_csv(core_comparison_path)
    optimization = pd.read_csv(optimization_path)
    recommendations = pd.read_csv(recommendations_path)
    walk_forward_folds = _optional_csv(walk_forward_folds_path)
    walk_forward_metrics = _optional_json(walk_forward_metrics_path)
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
        *_walk_forward_section(walk_forward_folds, walk_forward_metrics),
        "## Interpretation Boundaries",
        "",
        "- The live data window is short and uses a present-day static universe.",
        "- Nasdaq close is used as adjusted close; corporate actions may distort "
        "returns.",
        "- Optimizer results are recomputed on a common period before ranking.",
        "- Execution lag 1/2 sensitivity is included in recommendation robustness.",
        "- Candidate rankings describe historical fit under configured constraints.",
        "- Walk-forward uses a frozen strategy and does not refit factors inside "
        "each fold.",
        "- The current stitched OOS result has only three short folds; one strong "
        "fold dominates the aggregate.",
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


def _walk_forward_section(
    folds: pd.DataFrame | None,
    metrics: dict[str, float] | None,
) -> list[str]:
    if folds is None or metrics is None:
        return []
    return [
        "## Frozen-Strategy Walk-Forward OOS",
        "",
        f"- Folds: {len(folds)}",
        f"- Total return: {metrics['total_return']:.4f}",
        f"- Sharpe ratio: {metrics['sharpe_ratio']:.4f}",
        f"- Maximum drawdown: {metrics['max_drawdown']:.4f}",
        "",
        _markdown_table(
            folds,
            [
                "fold",
                "train_start",
                "train_end",
                "test_start",
                "test_end",
                "total_return",
                "sharpe_ratio",
                "max_drawdown",
            ],
        ),
        "",
    ]


def _optional_csv(path: str | Path) -> pd.DataFrame | None:
    source = Path(path)
    return pd.read_csv(source) if source.exists() else None


def _optional_json(path: str | Path) -> dict[str, float] | None:
    source = Path(path)
    if not source.exists():
        return None
    return json.loads(source.read_text(encoding="utf-8"))
