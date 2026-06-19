from __future__ import annotations

import numpy as np
import pandas as pd

from equity_transformer.factors.registry import FactorSpec


def factor_coverage(frame: pd.DataFrame, specs: list[FactorSpec]) -> pd.DataFrame:
    rows = []
    for spec in specs:
        values = pd.to_numeric(frame[spec.name], errors="coerce")
        rows.append(
            {
                "factor": spec.name,
                "family": spec.family,
                "direction": spec.direction,
                "coverage": float(values.notna().mean()),
                "missing_count": int(values.isna().sum()),
                "unique_values": int(values.nunique(dropna=True)),
            }
        )
    return pd.DataFrame(rows).sort_values(["family", "factor"])


def daily_information_coefficients(
    frame: pd.DataFrame,
    specs: list[FactorSpec],
    target_column: str,
    min_cross_section: int,
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        columns = ["date", spec.name, target_column]
        for date, daily in frame[columns].dropna().groupby("date"):
            if len(daily) < min_cross_section:
                continue
            if daily[spec.name].nunique() < 2 or daily[target_column].nunique() < 2:
                continue
            signal = daily[spec.name] * spec.direction
            pearson = signal.corr(daily[target_column], method="pearson")
            spearman = signal.corr(daily[target_column], method="spearman")
            if pd.notna(pearson) or pd.notna(spearman):
                rows.append(
                    {
                        "date": date,
                        "factor": spec.name,
                        "family": spec.family,
                        "ic": pearson,
                        "rank_ic": spearman,
                        "observations": len(daily),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=["date", "factor", "family", "ic", "rank_ic", "observations"]
        )
    return pd.DataFrame(rows).sort_values(["factor", "date"])


def summarize_information_coefficients(daily_ic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (factor, family), group in daily_ic.groupby(["factor", "family"]):
        ic_values = group["ic"].dropna()
        rank_values = group["rank_ic"].dropna()
        rows.append(
            {
                "factor": factor,
                "family": family,
                "mean_ic": _mean_or_nan(ic_values),
                "mean_rank_ic": _mean_or_nan(rank_values),
                "ic_std": _std_or_nan(ic_values),
                "rank_ic_std": _std_or_nan(rank_values),
                "icir": _information_ratio(ic_values),
                "rank_icir": _information_ratio(rank_values),
                "positive_ic_rate": _positive_rate(ic_values),
                "periods": int(len(group)),
                "mean_observations": float(group["observations"].mean()),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "factor",
                "family",
                "mean_ic",
                "mean_rank_ic",
                "ic_std",
                "rank_ic_std",
                "icir",
                "rank_icir",
                "positive_ic_rate",
                "periods",
                "mean_observations",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_rank_ic", "mean_ic"], ascending=False
    )


def quantile_forward_returns(
    frame: pd.DataFrame,
    specs: list[FactorSpec],
    target_column: str,
    quantiles: int,
    min_cross_section: int,
) -> pd.DataFrame:
    rows = []
    for spec in specs:
        columns = ["date", spec.name, target_column]
        for date, daily in frame[columns].dropna().groupby("date"):
            if len(daily) < max(min_cross_section, quantiles):
                continue
            signal = daily[spec.name] * spec.direction
            if signal.nunique() < quantiles:
                continue
            try:
                labels = pd.qcut(signal, quantiles, labels=False, duplicates="drop")
            except ValueError:
                continue
            grouped = daily.assign(quantile=labels.astype(int) + 1).groupby("quantile")
            for quantile, bucket in grouped:
                rows.append(
                    {
                        "date": date,
                        "factor": spec.name,
                        "family": spec.family,
                        "quantile": int(quantile),
                        "mean_forward_return": float(bucket[target_column].mean()),
                        "count": int(len(bucket)),
                    }
                )
    if not rows:
        return pd.DataFrame(
            columns=[
                "date",
                "factor",
                "family",
                "quantile",
                "mean_forward_return",
                "count",
            ]
        )
    return pd.DataFrame(rows).sort_values(["factor", "date", "quantile"])


def summarize_quantile_returns(quantile_returns: pd.DataFrame) -> pd.DataFrame:
    if quantile_returns.empty:
        return pd.DataFrame(
            columns=[
                "factor",
                "family",
                "quantile",
                "mean_forward_return",
                "periods",
                "mean_count",
            ]
        )
    return (
        quantile_returns.groupby(["factor", "family", "quantile"], as_index=False)
        .agg(
            mean_forward_return=("mean_forward_return", "mean"),
            periods=("mean_forward_return", "size"),
            mean_count=("count", "mean"),
        )
        .sort_values(["factor", "quantile"])
    )


def _mean_or_nan(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else float("nan")


def _std_or_nan(values: pd.Series) -> float:
    return float(values.std(ddof=1)) if len(values) > 1 else float("nan")


def _information_ratio(values: pd.Series) -> float:
    if len(values) < 2:
        return float("nan")
    std = values.std(ddof=1)
    return float(values.mean() / std) if std > 0 else float("nan")


def _positive_rate(values: pd.Series) -> float:
    return float(np.mean(values > 0)) if len(values) else float("nan")
