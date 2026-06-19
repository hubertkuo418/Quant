from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RandomForestConfig:
    n_estimators: int
    max_depth: int | None
    min_samples_leaf: int


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    subsample: float
    colsample_bytree: float


@dataclass(frozen=True)
class LightGBMConfig:
    n_estimators: int
    max_depth: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int


@dataclass(frozen=True)
class MLPConfig:
    hidden_dims: tuple[int, ...]
    dropout: float
    learning_rate: float
    weight_decay: float
    epochs: int
    patience: int
    batch_size: int


@dataclass(frozen=True)
class BaselineConfig:
    dataset_config: Path
    artifacts_dir: Path
    random_seed: int
    ridge_alpha: float
    momentum_feature: str
    random_forest: RandomForestConfig
    xgboost: XGBoostConfig
    lightgbm: LightGBMConfig
    mlp: MLPConfig


def load_baseline_config(path: str | Path) -> BaselineConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    xgb = payload["xgboost"]
    lgbm = payload["lightgbm"]
    random_forest = payload["random_forest"]
    mlp = payload["mlp"]
    return BaselineConfig(
        dataset_config=Path(payload["dataset_config"]),
        artifacts_dir=Path(payload["artifacts_dir"]),
        random_seed=int(payload["random_seed"]),
        ridge_alpha=float(payload["ridge_alpha"]),
        momentum_feature=str(payload["momentum_feature"]),
        random_forest=RandomForestConfig(
            n_estimators=int(random_forest["n_estimators"]),
            max_depth=(
                int(random_forest["max_depth"])
                if random_forest.get("max_depth") is not None
                else None
            ),
            min_samples_leaf=int(random_forest["min_samples_leaf"]),
        ),
        xgboost=XGBoostConfig(
            n_estimators=int(xgb["n_estimators"]),
            max_depth=int(xgb["max_depth"]),
            learning_rate=float(xgb["learning_rate"]),
            subsample=float(xgb["subsample"]),
            colsample_bytree=float(xgb["colsample_bytree"]),
        ),
        lightgbm=LightGBMConfig(
            n_estimators=int(lgbm["n_estimators"]),
            max_depth=int(lgbm["max_depth"]),
            learning_rate=float(lgbm["learning_rate"]),
            num_leaves=int(lgbm["num_leaves"]),
            min_child_samples=int(lgbm["min_child_samples"]),
        ),
        mlp=MLPConfig(
            hidden_dims=tuple(int(value) for value in mlp["hidden_dims"]),
            dropout=float(mlp["dropout"]),
            learning_rate=float(mlp["learning_rate"]),
            weight_decay=float(mlp["weight_decay"]),
            epochs=int(mlp["epochs"]),
            patience=int(mlp["patience"]),
            batch_size=int(mlp["batch_size"]),
        ),
    )
