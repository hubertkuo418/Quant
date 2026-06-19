from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ConfigEntry:
    name: str
    path: Path
    description: str


CONFIG_CATALOG: dict[str, ConfigEntry] = {
    "data": ConfigEntry("data", Path("configs/data.yaml"), "市場資料管線。"),
    "csv_import": ConfigEntry(
        "csv_import",
        Path("configs/csv_import.yaml"),
        "匯入本地市場資料 CSV。",
    ),
    "data_quality": ConfigEntry(
        "data_quality",
        Path("configs/data_quality.yaml"),
        "市場資料面板品質診斷。",
    ),
    "features": ConfigEntry(
        "features", Path("configs/features.yaml"), "符合時點原則的特徵。"
    ),
    "alphas": ConfigEntry(
        "alphas", Path("configs/alphas.yaml"), "公式化 Alpha 計算。"
    ),
    "factor_panel": ConfigEntry(
        "factor_panel", Path("configs/factor_panel.yaml"), "合併特徵與 Alpha。"
    ),
    "catalog": ConfigEntry(
        "catalog", Path("configs/catalog.yaml"), "DuckDB 研究資料目錄。"
    ),
    "factors": ConfigEntry(
        "factors", Path("configs/factors.yaml"), "因子驗證。"
    ),
    "factor_selection": ConfigEntry(
        "factor_selection",
        Path("configs/factor_selection.yaml"),
        "篩選通過驗證的因子。",
    ),
    "factor_signals": ConfigEntry(
        "factor_signals",
        Path("configs/factor_signals.yaml"),
        "產生 IC 加權因子訊號。",
    ),
    "strategy": ConfigEntry(
        "strategy", Path("configs/strategy.yaml"), "訊號轉換為權重的規則。"
    ),
    "model_strategy": ConfigEntry(
        "model_strategy",
        Path("configs/model_strategy.yaml"),
        "模型訊號轉換為權重的規則。",
    ),
    "backtest": ConfigEntry(
        "backtest", Path("configs/backtest.yaml"), "回測假設。"
    ),
    "model_backtest": ConfigEntry(
        "model_backtest",
        Path("configs/model_backtest.yaml"),
        "模型投資組合回測假設。",
    ),
    "benchmarks": ConfigEntry(
        "benchmarks", Path("configs/benchmarks.yaml"), "基準策略回測。"
    ),
    "regime": ConfigEntry(
        "regime", Path("configs/regime.yaml"), "市場狀態分析。"
    ),
    "sensitivity": ConfigEntry(
        "sensitivity",
        Path("configs/sensitivity.yaml"),
        "回測敏感度情境。",
    ),
    "attribution": ConfigEntry(
        "attribution",
        Path("configs/attribution.yaml"),
        "投資組合報酬歸因。",
    ),
    "equity_comparison": ConfigEntry(
        "equity_comparison",
        Path("configs/equity_comparison.yaml"),
        "共同期間投資組合比較。",
    ),
    "dataset": ConfigEntry(
        "dataset", Path("configs/dataset.yaml"), "建立機器學習資料集。"
    ),
    "baselines": ConfigEntry(
        "baselines", Path("configs/baselines.yaml"), "基準模型實驗。"
    ),
    "recurrent": ConfigEntry(
        "recurrent", Path("configs/recurrent.yaml"), "RNN/LSTM/GRU 訓練。"
    ),
    "transformer": ConfigEntry(
        "transformer", Path("configs/transformer.yaml"), "Transformer 訓練。"
    ),
    "prediction_signals": ConfigEntry(
        "prediction_signals",
        Path("configs/prediction_signals.yaml"),
        "將模型預測轉換為策略訊號。",
    ),
    "reporting": ConfigEntry(
        "reporting", Path("configs/reporting.yaml"), "彙整成果報告。"
    ),
    "studio_optimizer": ConfigEntry(
        "studio_optimizer",
        Path("configs/studio_optimizer.yaml"),
        "策略工作室參數搜尋與限制條件。",
    ),
    "studio_profile": ConfigEntry(
        "studio_profile",
        Path("configs/studio_profile.yaml"),
        "個人風險、報酬、回撤與換手率偏好。",
    ),
    "studio_walk_forward": ConfigEntry(
        "studio_walk_forward",
        Path("configs/studio_walk_forward.yaml"),
        "策略 Walk-forward 與滾動樣本外評估。",
    ),
}


def list_config_entries() -> list[ConfigEntry]:
    return list(CONFIG_CATALOG.values())


def read_config(config_name: str, root: str | Path = ".") -> dict[str, Any]:
    path = _config_path(config_name, root)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    with path.open(encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)
    return loaded or {}


def read_config_text(config_name: str, root: str | Path = ".") -> str:
    path = _config_path(config_name, root)
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    return path.read_text(encoding="utf-8")


def write_config_text(
    config_name: str,
    text: str,
    root: str | Path = ".",
) -> dict[str, Any]:
    parsed = yaml.safe_load(text)
    if parsed is None:
        parsed = {}
    if not isinstance(parsed, dict):
        raise ValueError("Config YAML must parse to a mapping/object.")
    path = _config_path(config_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return parsed


def _config_path(config_name: str, root: str | Path) -> Path:
    if config_name not in CONFIG_CATALOG:
        raise ValueError(f"Unknown config: {config_name}")
    return Path(root) / CONFIG_CATALOG[config_name].path
