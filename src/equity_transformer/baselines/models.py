from __future__ import annotations

import copy
import random

import numpy as np
import pandas as pd
import torch
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from xgboost import XGBRegressor

from equity_transformer.baselines.config import (
    BaselineConfig,
    MLPConfig,
)


def fit_historical_mean(y_train: np.ndarray, rows: int) -> np.ndarray:
    return np.tile(y_train.mean(axis=0), (rows, 1))


def fit_momentum(
    x_train_last: np.ndarray,
    y_train: np.ndarray,
    x_test_last: np.ndarray,
    feature_index: int,
    alpha: float,
) -> np.ndarray:
    feature_train = x_train_last[:, [feature_index]]
    feature_test = x_test_last[:, [feature_index]]
    return _fit_ridge_heads(feature_train, y_train, feature_test, alpha)


def fit_ridge(
    x_train_last: np.ndarray,
    y_train: np.ndarray,
    x_test_last: np.ndarray,
    alpha: float,
) -> np.ndarray:
    return _fit_ridge_heads(x_train_last, y_train, x_test_last, alpha)


def fit_random_forest(
    x_train_last: np.ndarray,
    y_train: np.ndarray,
    x_test_last: np.ndarray,
    config: BaselineConfig,
) -> np.ndarray:
    predictions = []
    for horizon_index in range(y_train.shape[1]):
        model = RandomForestRegressor(
            n_estimators=config.random_forest.n_estimators,
            max_depth=config.random_forest.max_depth,
            min_samples_leaf=config.random_forest.min_samples_leaf,
            random_state=config.random_seed,
            n_jobs=1,
        )
        model.fit(x_train_last, y_train[:, horizon_index])
        predictions.append(model.predict(x_test_last))
    return np.column_stack(predictions)


def _fit_ridge_heads(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> np.ndarray:
    predictions = []
    for horizon_index in range(y_train.shape[1]):
        model = Ridge(alpha=alpha)
        model.fit(x_train, y_train[:, horizon_index])
        predictions.append(model.predict(x_test))
    return np.column_stack(predictions)


def fit_xgboost(
    x_train_last: np.ndarray,
    y_train: np.ndarray,
    x_validation_last: np.ndarray,
    y_validation: np.ndarray,
    x_test_last: np.ndarray,
    config: BaselineConfig,
) -> np.ndarray:
    predictions = []
    for horizon_index in range(y_train.shape[1]):
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=config.xgboost.n_estimators,
            max_depth=config.xgboost.max_depth,
            learning_rate=config.xgboost.learning_rate,
            subsample=config.xgboost.subsample,
            colsample_bytree=config.xgboost.colsample_bytree,
            random_state=config.random_seed,
            n_jobs=1,
        )
        model.fit(
            x_train_last,
            y_train[:, horizon_index],
            eval_set=[(x_validation_last, y_validation[:, horizon_index])],
            verbose=False,
        )
        predictions.append(model.predict(x_test_last))
    return np.column_stack(predictions)


def fit_lightgbm(
    x_train_last: np.ndarray,
    y_train: np.ndarray,
    x_test_last: np.ndarray,
    config: BaselineConfig,
) -> np.ndarray:
    predictions = []
    columns = [f"feature_{index}" for index in range(x_train_last.shape[1])]
    train_frame = pd.DataFrame(x_train_last, columns=columns)
    test_frame = pd.DataFrame(x_test_last, columns=columns)
    for horizon_index in range(y_train.shape[1]):
        model = LGBMRegressor(
            objective="regression",
            n_estimators=config.lightgbm.n_estimators,
            max_depth=config.lightgbm.max_depth,
            learning_rate=config.lightgbm.learning_rate,
            num_leaves=config.lightgbm.num_leaves,
            min_child_samples=config.lightgbm.min_child_samples,
            random_state=config.random_seed,
            n_jobs=1,
            verbosity=-1,
        )
        target = np.ascontiguousarray(y_train[:, horizon_index])
        model.fit(train_frame, target)
        predictions.append(model.predict(test_frame))
    return np.column_stack(predictions)


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features.flatten(start_dim=1))


def fit_mlp(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    x_test: np.ndarray,
    config: MLPConfig,
    seed: int,
) -> np.ndarray:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = SimpleMLP(
        input_dim=x_train.shape[1] * x_train.shape[2],
        output_dim=y_train.shape[1],
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_function = nn.HuberLoss()
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=config.batch_size,
        shuffle=True,
    )
    validation_x = torch.from_numpy(x_validation)
    validation_y = torch.from_numpy(y_validation)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    stale_epochs = 0

    for _ in range(config.epochs):
        model.train()
        for features, targets in train_loader:
            optimizer.zero_grad()
            loss = loss_function(model(features), targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_loss = loss_function(model(validation_x), validation_y).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x_test)).numpy()
