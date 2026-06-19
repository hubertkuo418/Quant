from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class WorkflowStep:
    name: str
    label: str
    command: tuple[str, ...]
    description: str


WORKFLOW_STEPS: dict[str, WorkflowStep] = {
    "studio_strategy": WorkflowStep(
        name="studio_strategy",
        label="Run Studio Strategy",
        command=(
            "scripts/run_studio_strategy.py",
            "--spec",
            "strategies/factor_top10.yaml",
        ),
        description="Execute a versioned StrategySpec and register the run.",
    ),
    "studio_optimizer": WorkflowStep(
        name="studio_optimizer",
        label="Optimize Studio Strategy",
        command=(
            "scripts/optimize_studio_strategy.py",
            "--config",
            "configs/studio_optimizer.yaml",
        ),
        description="Search constraints and mark common-period Pareto candidates.",
    ),
    "studio_recommend": WorkflowStep(
        name="studio_recommend",
        label="Recommend Studio Candidates",
        command=(
            "scripts/recommend_studio_strategies.py",
            "--profile",
            "configs/studio_profile.yaml",
        ),
        description="Rank robust candidates for configured user constraints.",
    ),
    "studio_report": WorkflowStep(
        name="studio_report",
        label="Build Studio Report",
        command=("scripts/build_studio_report.py",),
        description="Generate a reproducible report from registered artifacts.",
    ),
    "csv_import": WorkflowStep(
        name="csv_import",
        label="Import Market CSV",
        command=("scripts/import_market_csv.py", "--config", "configs/csv_import.yaml"),
        description="Validate local OHLCV CSV files and build the market panel.",
    ),
    "data_quality": WorkflowStep(
        name="data_quality",
        label="Analyze Market Quality",
        command=(
            "scripts/analyze_market_quality.py",
            "--config",
            "configs/data_quality.yaml",
        ),
        description="Check ticker coverage, stale prices, volume, and return outliers.",
    ),
    "features": WorkflowStep(
        name="features",
        label="Build Features",
        command=("scripts/build_features.py", "--config", "configs/features.yaml"),
        description="Create point-in-time technical, fundamental, and news features.",
    ),
    "alphas": WorkflowStep(
        name="alphas",
        label="Build Alphas",
        command=("scripts/build_alphas.py", "--config", "configs/alphas.yaml"),
        description="Calculate registered formulaic alpha columns.",
    ),
    "factor_panel": WorkflowStep(
        name="factor_panel",
        label="Merge Factor Panel",
        command=(
            "scripts/build_factor_panel.py",
            "--config",
            "configs/factor_panel.yaml",
        ),
        description="Merge feature and alpha panels into one research table.",
    ),
    "catalog": WorkflowStep(
        name="catalog",
        label="Build DuckDB Catalog",
        command=("scripts/build_catalog.py", "--config", "configs/catalog.yaml"),
        description="Register generated Parquet artifacts as DuckDB views.",
    ),
    "validate_factors": WorkflowStep(
        name="validate_factors",
        label="Validate Factors",
        command=("scripts/validate_factors.py", "--config", "configs/factors.yaml"),
        description="Compute coverage, IC, Rank IC, and quantile returns.",
    ),
    "select_factors": WorkflowStep(
        name="select_factors",
        label="Select Factors",
        command=(
            "scripts/select_factors.py",
            "--config",
            "configs/factor_selection.yaml",
        ),
        description="Select validated factors from IC and coverage reports.",
    ),
    "factor_signals": WorkflowStep(
        name="factor_signals",
        label="Build Factor Signals",
        command=(
            "scripts/build_factor_signals.py",
            "--config",
            "configs/factor_signals.yaml",
        ),
        description="Blend selected factors into an IC-weighted signal.",
    ),
    "strategy": WorkflowStep(
        name="strategy",
        label="Build Strategy",
        command=("scripts/build_strategy.py", "--config", "configs/strategy.yaml"),
        description="Convert a score column into target portfolio weights.",
    ),
    "backtest": WorkflowStep(
        name="backtest",
        label="Run Backtest",
        command=("scripts/run_backtest.py", "--config", "configs/backtest.yaml"),
        description="Simulate NAV, holdings, turnover, and costs.",
    ),
    "benchmarks": WorkflowStep(
        name="benchmarks",
        label="Run Benchmarks",
        command=("scripts/run_benchmarks.py", "--config", "configs/benchmarks.yaml"),
        description="Run equal-weight and buy-and-hold benchmark backtests.",
    ),
    "regime_analysis": WorkflowStep(
        name="regime_analysis",
        label="Analyze Market Regimes",
        command=("scripts/analyze_regimes.py", "--config", "configs/regime.yaml"),
        description="Evaluate strategy performance across trend/volatility regimes.",
    ),
    "sensitivity": WorkflowStep(
        name="sensitivity",
        label="Run Sensitivity Analysis",
        command=(
            "scripts/run_sensitivity.py",
            "--config",
            "configs/sensitivity.yaml",
        ),
        description="Stress costs, execution lag, liquidity, and financing inputs.",
    ),
    "attribution": WorkflowStep(
        name="attribution",
        label="Build Return Attribution",
        command=(
            "scripts/build_attribution.py",
            "--config",
            "configs/attribution.yaml",
        ),
        description="Measure ticker contributions and portfolio concentration.",
    ),
    "equity_comparison": WorkflowStep(
        name="equity_comparison",
        label="Compare Equity Curves",
        command=(
            "scripts/compare_equity_curves.py",
            "--config",
            "configs/equity_comparison.yaml",
        ),
        description="Recompute factor, model, and benchmark metrics on common dates.",
    ),
    "dataset": WorkflowStep(
        name="dataset",
        label="Build ML Dataset",
        command=("scripts/build_dataset.py", "--config", "configs/dataset.yaml"),
        description="Create model-ready sequences with purged chronological splits.",
    ),
    "baselines": WorkflowStep(
        name="baselines",
        label="Run Baselines",
        command=("scripts/run_baselines.py", "--config", "configs/baselines.yaml"),
        description="Train and evaluate baseline forecasting models.",
    ),
    "transformer": WorkflowStep(
        name="transformer",
        label="Train Transformer",
        command=(
            "scripts/train_transformer.py",
            "--config",
            "configs/transformer.yaml",
        ),
        description="Train the Transformer forecasting model.",
    ),
    "prediction_signals": WorkflowStep(
        name="prediction_signals",
        label="Build Prediction Signals",
        command=(
            "scripts/build_prediction_signals.py",
            "--config",
            "configs/prediction_signals.yaml",
        ),
        description="Convert Transformer predictions into strategy-ready signals.",
    ),
    "model_strategy": WorkflowStep(
        name="model_strategy",
        label="Build Model Strategy",
        command=(
            "scripts/build_strategy.py",
            "--config",
            "configs/model_strategy.yaml",
        ),
        description="Rank model predictions into dedicated target weights.",
    ),
    "model_backtest": WorkflowStep(
        name="model_backtest",
        label="Run Model Backtest",
        command=(
            "scripts/run_backtest.py",
            "--config",
            "configs/model_backtest.yaml",
        ),
        description="Backtest model-driven weights with execution lag and SPY.",
    ),
    "recurrent": WorkflowStep(
        name="recurrent",
        label="Train RNN/LSTM/GRU",
        command=("scripts/train_recurrent.py", "--config", "configs/recurrent.yaml"),
        description="Train a recurrent sequence baseline.",
    ),
}


def list_workflow_steps() -> list[WorkflowStep]:
    return list(WORKFLOW_STEPS.values())


def run_workflow_step(
    step_name: str,
    root: str | Path = ".",
    timeout_seconds: int = 600,
) -> dict[str, object]:
    if step_name not in WORKFLOW_STEPS:
        raise ValueError(f"Unknown workflow step: {step_name}")
    root_path = Path(root)
    step = WORKFLOW_STEPS[step_name]
    started = datetime.now(UTC)
    command = [sys.executable, *step.command]
    completed = subprocess.run(
        command,
        cwd=root_path,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    finished = datetime.now(UTC)
    result = {
        "step": step.name,
        "label": step.label,
        "command": command,
        "returncode": completed.returncode,
        "started_utc": started.isoformat(),
        "finished_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "success": completed.returncode == 0,
    }
    _write_run_record(root_path, result)
    return result


def latest_workflow_runs(
    root: str | Path = ".", limit: int = 20
) -> list[dict[str, object]]:
    records_dir = Path(root) / "artifacts" / "workflow_runs"
    if not records_dir.exists():
        return []
    records = []
    for path in sorted(records_dir.glob("*.json"), reverse=True)[:limit]:
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _write_run_record(root: Path, result: dict[str, object]) -> None:
    records_dir = root / "artifacts" / "workflow_runs"
    records_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = records_dir / f"{timestamp}_{result['step']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
