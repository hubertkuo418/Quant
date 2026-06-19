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
    "data": ConfigEntry("data", Path("configs/data.yaml"), "Market data pipeline."),
    "csv_import": ConfigEntry(
        "csv_import",
        Path("configs/csv_import.yaml"),
        "Local CSV market data import.",
    ),
    "data_quality": ConfigEntry(
        "data_quality",
        Path("configs/data_quality.yaml"),
        "Market panel quality diagnostics.",
    ),
    "features": ConfigEntry(
        "features", Path("configs/features.yaml"), "Point-in-time features."
    ),
    "alphas": ConfigEntry(
        "alphas", Path("configs/alphas.yaml"), "Formulaic alpha calculation."
    ),
    "factor_panel": ConfigEntry(
        "factor_panel", Path("configs/factor_panel.yaml"), "Feature/alpha merge."
    ),
    "catalog": ConfigEntry(
        "catalog", Path("configs/catalog.yaml"), "DuckDB research catalog."
    ),
    "factors": ConfigEntry(
        "factors", Path("configs/factors.yaml"), "Factor validation."
    ),
    "factor_selection": ConfigEntry(
        "factor_selection",
        Path("configs/factor_selection.yaml"),
        "Validated factor selection.",
    ),
    "factor_signals": ConfigEntry(
        "factor_signals",
        Path("configs/factor_signals.yaml"),
        "IC-weighted factor signal generation.",
    ),
    "strategy": ConfigEntry(
        "strategy", Path("configs/strategy.yaml"), "Signal-to-weight rules."
    ),
    "model_strategy": ConfigEntry(
        "model_strategy",
        Path("configs/model_strategy.yaml"),
        "Model signal-to-weight rules.",
    ),
    "backtest": ConfigEntry(
        "backtest", Path("configs/backtest.yaml"), "Backtest assumptions."
    ),
    "model_backtest": ConfigEntry(
        "model_backtest",
        Path("configs/model_backtest.yaml"),
        "Model portfolio backtest assumptions.",
    ),
    "benchmarks": ConfigEntry(
        "benchmarks", Path("configs/benchmarks.yaml"), "Benchmark backtests."
    ),
    "regime": ConfigEntry(
        "regime", Path("configs/regime.yaml"), "Market regime analysis."
    ),
    "sensitivity": ConfigEntry(
        "sensitivity",
        Path("configs/sensitivity.yaml"),
        "Backtest sensitivity scenarios.",
    ),
    "attribution": ConfigEntry(
        "attribution",
        Path("configs/attribution.yaml"),
        "Portfolio return attribution.",
    ),
    "equity_comparison": ConfigEntry(
        "equity_comparison",
        Path("configs/equity_comparison.yaml"),
        "Common-period portfolio comparison.",
    ),
    "dataset": ConfigEntry(
        "dataset", Path("configs/dataset.yaml"), "ML dataset construction."
    ),
    "baselines": ConfigEntry(
        "baselines", Path("configs/baselines.yaml"), "Baseline model experiments."
    ),
    "recurrent": ConfigEntry(
        "recurrent", Path("configs/recurrent.yaml"), "RNN/LSTM/GRU training."
    ),
    "transformer": ConfigEntry(
        "transformer", Path("configs/transformer.yaml"), "Transformer training."
    ),
    "prediction_signals": ConfigEntry(
        "prediction_signals",
        Path("configs/prediction_signals.yaml"),
        "Model prediction-to-signal conversion.",
    ),
    "reporting": ConfigEntry(
        "reporting", Path("configs/reporting.yaml"), "Report aggregation."
    ),
    "studio_optimizer": ConfigEntry(
        "studio_optimizer",
        Path("configs/studio_optimizer.yaml"),
        "Strategy Studio parameter search and constraints.",
    ),
    "studio_profile": ConfigEntry(
        "studio_profile",
        Path("configs/studio_profile.yaml"),
        "Personal risk, return, drawdown, and turnover preferences.",
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
