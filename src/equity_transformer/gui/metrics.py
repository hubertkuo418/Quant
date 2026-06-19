from __future__ import annotations

from collections.abc import Iterable, Mapping

import pandas as pd

OVERVIEW_METRICS = (
    "annual_return",
    "sharpe_ratio",
    "max_drawdown",
    "average_turnover",
)

TAIL_RISK_METRICS = (
    "value_at_risk_95",
    "conditional_value_at_risk_95",
    "profit_factor",
    "sortino_ratio",
    "calmar_ratio",
)

RELATIVE_METRICS = (
    "information_ratio",
    "tracking_error",
    "beta",
    "annual_alpha",
)

FINANCING_METRICS = (
    "total_cost",
    "total_cash_interest",
    "total_borrow_cost",
)


def available_metrics(
    metrics: Mapping[str, object], keys: Iterable[str]
) -> list[tuple[str, float]]:
    result = []
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, int | float):
            result.append((key, float(value)))
    return result


def metric_table(metrics: Mapping[str, object], keys: Iterable[str]) -> pd.DataFrame:
    rows = [
        {"metric": key.replace("_", " ").title(), "value": value}
        for key, value in available_metrics(metrics, keys)
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])
