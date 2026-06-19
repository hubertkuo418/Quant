from __future__ import annotations

import numpy as np
import pandas as pd


def build_target_weights(
    signals: pd.DataFrame,
    date_column: str,
    ticker_column: str,
    score_column: str,
    strategy_type: str,
    top_k: int = 10,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    weighting: str = "equal",
    rebalance_frequency: str = "W-FRI",
    volatility_column: str | None = None,
    sector_column: str | None = None,
    max_sector_weight: float | None = None,
    max_position_weight: float | None = None,
) -> pd.DataFrame:
    columns = [date_column, ticker_column, score_column]
    if volatility_column:
        columns.append(volatility_column)
    if sector_column:
        columns.append(sector_column)
    prepared = signals[columns].copy()
    prepared[date_column] = pd.to_datetime(prepared[date_column])
    prepared = prepared.dropna(subset=[score_column])
    if weighting in {"inverse_volatility", "risk_parity"}:
        if volatility_column is None:
            raise ValueError(f"{weighting} weighting requires volatility_column.")
        prepared = prepared.dropna(subset=[volatility_column])
    rebalance_dates = _rebalance_dates(prepared[date_column], rebalance_frequency)
    prepared = prepared[prepared[date_column].isin(rebalance_dates)]

    parts = []
    for date, daily in prepared.groupby(date_column, sort=True):
        daily = daily.sort_values(score_column, ascending=False).reset_index(drop=True)
        if strategy_type == "long_only_top_k":
            weights = _long_only_top_k(
                daily,
                ticker_column,
                score_column,
                top_k,
                weighting,
                volatility_column,
                sector_column,
            )
        elif strategy_type == "long_short_quantile":
            weights = _long_short_quantile(
                daily,
                ticker_column,
                score_column,
                long_quantile,
                short_quantile,
                weighting,
                volatility_column,
                sector_column,
            )
        else:
            raise ValueError(f"Unsupported strategy_type: {strategy_type}")
        if max_position_weight is not None:
            weights = apply_position_cap(weights, max_position_weight)
        if sector_column and max_sector_weight is not None:
            weights = apply_sector_cap(weights, max_sector_weight)
        weights.insert(0, "date", date)
        parts.append(weights)

    if not parts:
        return pd.DataFrame(columns=["date", "ticker", "weight", "side", "score"])
    return pd.concat(parts, ignore_index=True).sort_values(["date", "ticker"])


def _rebalance_dates(dates: pd.Series, frequency: str) -> set[pd.Timestamp]:
    unique_dates = pd.Series(pd.to_datetime(dates).drop_duplicates()).sort_values()
    if unique_dates.empty:
        return set()
    if frequency.lower() in {"daily", "d"}:
        return set(unique_dates)
    grouped = unique_dates.groupby(unique_dates.dt.to_period(frequency))
    return set(grouped.max())


def _long_only_top_k(
    daily: pd.DataFrame,
    ticker_column: str,
    score_column: str,
    top_k: int,
    weighting: str,
    volatility_column: str | None,
    sector_column: str | None,
) -> pd.DataFrame:
    selected = daily.nlargest(top_k, score_column).copy()
    selected["weight"] = _positive_weights(
        selected[score_column],
        weighting,
        selected[volatility_column] if volatility_column else None,
    )
    selected["side"] = "long"
    renamed = selected.rename(
        columns={ticker_column: "ticker", score_column: "score"}
    )
    columns = ["ticker", "weight", "side", "score"]
    if sector_column:
        renamed = renamed.rename(columns={sector_column: "sector"})
        columns.append("sector")
    return renamed[columns]


def _long_short_quantile(
    daily: pd.DataFrame,
    ticker_column: str,
    score_column: str,
    long_quantile: float,
    short_quantile: float,
    weighting: str,
    volatility_column: str | None,
    sector_column: str | None,
) -> pd.DataFrame:
    count = len(daily)
    long_count = max(1, int(np.floor(count * long_quantile)))
    short_count = max(1, int(np.floor(count * short_quantile)))
    longs = daily.nlargest(long_count, score_column).copy()
    shorts = daily.nsmallest(short_count, score_column).copy()

    long_risk = longs[volatility_column] if volatility_column else None
    short_risk = shorts[volatility_column] if volatility_column else None
    longs["weight"] = 0.5 * _positive_weights(
        longs[score_column],
        weighting,
        long_risk,
    )
    shorts["weight"] = -0.5 * _positive_weights(
        -shorts[score_column],
        weighting,
        short_risk,
    )
    longs["side"] = "long"
    shorts["side"] = "short"
    result = pd.concat([longs, shorts], ignore_index=True)
    renamed = result.rename(
        columns={ticker_column: "ticker", score_column: "score"}
    )
    columns = ["ticker", "weight", "side", "score"]
    if sector_column:
        renamed = renamed.rename(columns={sector_column: "sector"})
        columns.append("sector")
    return renamed[columns]


def _positive_weights(
    scores: pd.Series,
    weighting: str,
    volatility: pd.Series | None = None,
) -> np.ndarray:
    if len(scores) == 0:
        return np.array([], dtype=float)
    if weighting == "equal":
        return np.full(len(scores), 1 / len(scores), dtype=float)
    if weighting == "score":
        shifted = scores - scores.min()
        if float(shifted.sum()) <= 0:
            return np.full(len(scores), 1 / len(scores), dtype=float)
        return (shifted / shifted.sum()).to_numpy(dtype=float)
    if weighting in {"inverse_volatility", "risk_parity"}:
        if volatility is None:
            raise ValueError(f"{weighting} weighting requires volatility values.")
        clean = volatility.astype(float).replace([np.inf, -np.inf], np.nan)
        clean = clean.where(clean > 0)
        if clean.isna().any():
            raise ValueError(f"{weighting} weighting requires positive volatility.")
        inverse = 1 / clean
        return (inverse / inverse.sum()).to_numpy(dtype=float)
    raise ValueError(f"Unsupported weighting: {weighting}")


def apply_sector_cap(weights: pd.DataFrame, max_sector_weight: float) -> pd.DataFrame:
    if "sector" not in weights.columns:
        raise ValueError("Sector cap requires a sector column in weights.")
    if max_sector_weight <= 0:
        raise ValueError("max_sector_weight must be positive.")
    capped = weights.copy()

    for _sector, sector_rows in capped.groupby("sector"):
        gross = sector_rows["weight"].abs().sum()
        if gross > max_sector_weight:
            scale = max_sector_weight / gross
            capped.loc[sector_rows.index, "weight"] *= scale
    return capped


def apply_position_cap(
    weights: pd.DataFrame, max_position_weight: float
) -> pd.DataFrame:
    if max_position_weight <= 0:
        raise ValueError("max_position_weight must be positive.")
    capped = weights.copy()
    capped["weight"] = capped["weight"].clip(
        lower=-max_position_weight,
        upper=max_position_weight,
    )
    return capped
