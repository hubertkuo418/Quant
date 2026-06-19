from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    actual = predictions["target"].to_numpy()
    predicted = predictions["prediction"].to_numpy()
    error = predicted - actual

    daily_pearson = _daily_correlation(predictions, method="pearson")
    daily_spearman = _daily_correlation(predictions, method="spearman")
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson_ic": daily_pearson,
        "rank_ic": daily_spearman,
        "directional_accuracy": float(np.mean((predicted >= 0) == (actual >= 0))),
    }


def _daily_correlation(frame: pd.DataFrame, method: str) -> float:
    correlations = []
    for _, daily in frame.groupby("date"):
        if len(daily) < 2:
            continue
        if daily["prediction"].nunique() < 2 or daily["target"].nunique() < 2:
            continue
        value = daily["prediction"].corr(daily["target"], method=method)
        if pd.notna(value):
            correlations.append(value)
    return float(np.mean(correlations)) if correlations else float("nan")
