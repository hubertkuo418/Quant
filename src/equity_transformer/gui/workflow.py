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
        label="執行工作室策略",
        command=(
            "scripts/run_studio_strategy.py",
            "--spec",
            "strategies/factor_top10.yaml",
        ),
        description="執行版本化策略規格並登錄本次結果。",
    ),
    "studio_optimizer": WorkflowStep(
        name="studio_optimizer",
        label="優化工作室策略",
        command=(
            "scripts/optimize_studio_strategy.py",
            "--config",
            "configs/studio_optimizer.yaml",
        ),
        description="搜尋參數並標示共同期間的 Pareto 候選方案。",
    ),
    "studio_recommend": WorkflowStep(
        name="studio_recommend",
        label="推薦策略候選方案",
        command=(
            "scripts/recommend_studio_strategies.py",
            "--profile",
            "configs/studio_profile.yaml",
        ),
        description="依使用者限制條件排序穩健的候選方案。",
    ),
    "studio_report": WorkflowStep(
        name="studio_report",
        label="建立工作室報告",
        command=("scripts/build_studio_report.py",),
        description="根據已登錄產物建立可重現的成果報告。",
    ),
    "csv_import": WorkflowStep(
        name="csv_import",
        label="匯入市場 CSV",
        command=("scripts/import_market_csv.py", "--config", "configs/csv_import.yaml"),
        description="驗證本地 OHLCV CSV 並建立市場資料面板。",
    ),
    "data_quality": WorkflowStep(
        name="data_quality",
        label="分析市場資料品質",
        command=(
            "scripts/analyze_market_quality.py",
            "--config",
            "configs/data_quality.yaml",
        ),
        description="檢查股票覆蓋率、價格停滯、成交量與異常報酬。",
    ),
    "features": WorkflowStep(
        name="features",
        label="建立特徵",
        command=("scripts/build_features.py", "--config", "configs/features.yaml"),
        description="建立符合時點原則的技術、基本面與新聞特徵。",
    ),
    "alphas": WorkflowStep(
        name="alphas",
        label="建立 Alpha 因子",
        command=("scripts/build_alphas.py", "--config", "configs/alphas.yaml"),
        description="計算已登錄的公式化 Alpha 欄位。",
    ),
    "factor_panel": WorkflowStep(
        name="factor_panel",
        label="合併因子面板",
        command=(
            "scripts/build_factor_panel.py",
            "--config",
            "configs/factor_panel.yaml",
        ),
        description="將特徵與 Alpha 面板合併為研究資料表。",
    ),
    "catalog": WorkflowStep(
        name="catalog",
        label="建立 DuckDB 目錄",
        command=("scripts/build_catalog.py", "--config", "configs/catalog.yaml"),
        description="將產生的 Parquet 研究產物登錄為 DuckDB View。",
    ),
    "validate_factors": WorkflowStep(
        name="validate_factors",
        label="驗證因子",
        command=("scripts/validate_factors.py", "--config", "configs/factors.yaml"),
        description="計算覆蓋率、IC、Rank IC 與分位數報酬。",
    ),
    "select_factors": WorkflowStep(
        name="select_factors",
        label="篩選因子",
        command=(
            "scripts/select_factors.py",
            "--config",
            "configs/factor_selection.yaml",
        ),
        description="根據 IC 與覆蓋率報告篩選通過驗證的因子。",
    ),
    "factor_signals": WorkflowStep(
        name="factor_signals",
        label="建立因子訊號",
        command=(
            "scripts/build_factor_signals.py",
            "--config",
            "configs/factor_signals.yaml",
        ),
        description="將已選因子合成 IC 加權訊號。",
    ),
    "strategy": WorkflowStep(
        name="strategy",
        label="建立策略",
        command=("scripts/build_strategy.py", "--config", "configs/strategy.yaml"),
        description="將分數欄位轉換為投資組合目標權重。",
    ),
    "backtest": WorkflowStep(
        name="backtest",
        label="執行回測",
        command=("scripts/run_backtest.py", "--config", "configs/backtest.yaml"),
        description="模擬淨值、持倉、換手率與交易成本。",
    ),
    "benchmarks": WorkflowStep(
        name="benchmarks",
        label="執行基準策略",
        command=("scripts/run_benchmarks.py", "--config", "configs/benchmarks.yaml"),
        description="執行等權重與買入持有基準回測。",
    ),
    "regime_analysis": WorkflowStep(
        name="regime_analysis",
        label="分析市場狀態",
        command=("scripts/analyze_regimes.py", "--config", "configs/regime.yaml"),
        description="評估策略在不同趨勢與波動狀態下的績效。",
    ),
    "sensitivity": WorkflowStep(
        name="sensitivity",
        label="執行敏感度分析",
        command=(
            "scripts/run_sensitivity.py",
            "--config",
            "configs/sensitivity.yaml",
        ),
        description="對成本、執行延遲、流動性與融資條件進行壓力測試。",
    ),
    "attribution": WorkflowStep(
        name="attribution",
        label="建立報酬歸因",
        command=(
            "scripts/build_attribution.py",
            "--config",
            "configs/attribution.yaml",
        ),
        description="衡量個股貢獻與投資組合集中度。",
    ),
    "equity_comparison": WorkflowStep(
        name="equity_comparison",
        label="比較淨值曲線",
        command=(
            "scripts/compare_equity_curves.py",
            "--config",
            "configs/equity_comparison.yaml",
        ),
        description="在共同日期重新計算因子、模型與基準績效。",
    ),
    "dataset": WorkflowStep(
        name="dataset",
        label="建立機器學習資料集",
        command=("scripts/build_dataset.py", "--config", "configs/dataset.yaml"),
        description="以清除重疊的時間切分建立模型序列資料。",
    ),
    "baselines": WorkflowStep(
        name="baselines",
        label="執行基準模型",
        command=("scripts/run_baselines.py", "--config", "configs/baselines.yaml"),
        description="訓練並評估基準預測模型。",
    ),
    "transformer": WorkflowStep(
        name="transformer",
        label="訓練 Transformer",
        command=(
            "scripts/train_transformer.py",
            "--config",
            "configs/transformer.yaml",
        ),
        description="訓練 Transformer 報酬預測模型。",
    ),
    "prediction_signals": WorkflowStep(
        name="prediction_signals",
        label="建立預測訊號",
        command=(
            "scripts/build_prediction_signals.py",
            "--config",
            "configs/prediction_signals.yaml",
        ),
        description="將 Transformer 預測轉換為可供策略使用的訊號。",
    ),
    "model_strategy": WorkflowStep(
        name="model_strategy",
        label="建立模型策略",
        command=(
            "scripts/build_strategy.py",
            "--config",
            "configs/model_strategy.yaml",
        ),
        description="將模型預測排序並轉換為專用目標權重。",
    ),
    "model_backtest": WorkflowStep(
        name="model_backtest",
        label="執行模型回測",
        command=(
            "scripts/run_backtest.py",
            "--config",
            "configs/model_backtest.yaml",
        ),
        description="使用執行延遲與 SPY 基準回測模型權重。",
    ),
    "recurrent": WorkflowStep(
        name="recurrent",
        label="訓練 RNN/LSTM/GRU",
        command=("scripts/train_recurrent.py", "--config", "configs/recurrent.yaml"),
        description="訓練循環神經網路序列基準模型。",
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
