from __future__ import annotations

import pandas as pd

FUNDAMENTAL_COLUMNS = ["pe_ratio", "pb_ratio"]


def merge_point_in_time_fundamentals(
    features: pd.DataFrame, fundamentals: pd.DataFrame
) -> pd.DataFrame:
    required = {"ticker", "available_at", *FUNDAMENTAL_COLUMNS}
    missing = required.difference(fundamentals.columns)
    if missing:
        raise ValueError(f"Missing fundamental columns: {sorted(missing)}")

    source = fundamentals.copy()
    source["ticker"] = source["ticker"].str.upper()
    source["available_at"] = pd.to_datetime(source["available_at"]).dt.normalize()
    if source[["ticker", "available_at"]].duplicated().any():
        raise ValueError("Duplicate (ticker, available_at) fundamental rows found.")

    parts: list[pd.DataFrame] = []
    for ticker, prices in features.groupby("ticker", sort=False):
        ticker_source = source[source["ticker"] == ticker].sort_values("available_at")
        ticker_prices = prices.sort_values("date")
        if ticker_source.empty:
            merged = ticker_prices.copy()
            for column in FUNDAMENTAL_COLUMNS:
                merged[column] = float("nan")
            merged["fundamental_available_at"] = pd.NaT
        else:
            merged = pd.merge_asof(
                ticker_prices,
                ticker_source[
                    ["available_at", *FUNDAMENTAL_COLUMNS]
                ].rename(columns={"available_at": "fundamental_available_at"}),
                left_on="date",
                right_on="fundamental_available_at",
                direction="backward",
                allow_exact_matches=True,
            )
        parts.append(merged)

    return (
        pd.concat(parts, ignore_index=True)
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )
