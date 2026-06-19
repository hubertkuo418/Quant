from __future__ import annotations

import pandas as pd


def add_forward_return_targets(
    frame: pd.DataFrame, horizons: tuple[int, ...]
) -> pd.DataFrame:
    result = frame.sort_values(["ticker", "date"]).reset_index(drop=True).copy()
    grouped = result.groupby("ticker", sort=False)

    for horizon in horizons:
        future_price = grouped["adj_close"].shift(-horizon)
        result[f"target_{horizon}d"] = future_price / result["adj_close"] - 1
        result[f"target_date_{horizon}d"] = grouped["date"].shift(-horizon)

    return result.sort_values(["date", "ticker"]).reset_index(drop=True)


def assign_purged_splits(
    frame: pd.DataFrame,
    horizons: tuple[int, ...],
    train_end: str,
    validation_end: str,
) -> pd.DataFrame:
    result = frame.copy()
    max_horizon = max(horizons)
    realization_date = result[f"target_date_{max_horizon}d"]
    feature_date = result["date"]
    train_cutoff = pd.Timestamp(train_end)
    validation_cutoff = pd.Timestamp(validation_end)

    result["split"] = "unassigned"
    result.loc[realization_date <= train_cutoff, "split"] = "train"
    result.loc[
        (feature_date > train_cutoff)
        & (realization_date <= validation_cutoff),
        "split",
    ] = "validation"
    result.loc[
        (feature_date > validation_cutoff) & realization_date.notna(),
        "split",
    ] = "test"
    return result
