from __future__ import annotations

from pathlib import Path

import pandas as pd

UNIVERSE_COLUMNS = ["ticker", "start_date", "end_date"]


def load_universe_membership(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError("Universe membership file is empty.")
    missing = set(UNIVERSE_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"Universe membership missing: {sorted(missing)}")
    result = frame[UNIVERSE_COLUMNS].copy()
    result["ticker"] = result["ticker"].astype(str).str.strip().str.upper()
    result["start_date"] = pd.to_datetime(result["start_date"], errors="coerce")
    result["end_date"] = pd.to_datetime(result["end_date"], errors="coerce")
    if result["ticker"].eq("").any() or result["start_date"].isna().any():
        raise ValueError("Universe ticker and start_date must be populated.")
    invalid = result["end_date"].notna() & (
        result["end_date"] < result["start_date"]
    )
    if invalid.any():
        raise ValueError("Universe end_date cannot precede start_date.")
    result = result.sort_values(["ticker", "start_date"]).reset_index(drop=True)
    _validate_non_overlapping_intervals(result)
    return result


def membership_tickers(membership: pd.DataFrame) -> tuple[str, ...]:
    return tuple(membership["ticker"].drop_duplicates().tolist())


def filter_market_to_membership(
    market: pd.DataFrame,
    membership: pd.DataFrame,
    always_include: tuple[str, ...] = (),
) -> pd.DataFrame:
    required = {"date", "ticker"}
    missing = required.difference(market.columns)
    if missing:
        raise ValueError(f"Market frame missing: {sorted(missing)}")
    source = market.copy()
    source["date"] = pd.to_datetime(source["date"])
    source["ticker"] = source["ticker"].astype(str).str.upper()
    source["_row_id"] = range(len(source))
    joined = source.merge(membership, on="ticker", how="left", validate="m:m")
    active = (joined["date"] >= joined["start_date"]) & (
        joined["end_date"].isna() | (joined["date"] <= joined["end_date"])
    )
    benchmark = joined["ticker"].isin(ticker.upper() for ticker in always_include)
    selected_ids = joined.loc[active | benchmark, "_row_id"].drop_duplicates()
    return (
        source.loc[source["_row_id"].isin(selected_ids)]
        .drop(columns="_row_id")
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )


def _validate_non_overlapping_intervals(frame: pd.DataFrame) -> None:
    for ticker, group in frame.groupby("ticker", sort=False):
        previous_end: pd.Timestamp | None = None
        previous_open_ended = False
        for row in group.itertuples(index=False):
            if previous_open_ended:
                raise ValueError(f"Open-ended universe interval overlaps for {ticker}.")
            if previous_end is not None and row.start_date <= previous_end:
                raise ValueError(f"Universe intervals overlap for {ticker}.")
            previous_open_ended = pd.isna(row.end_date)
            previous_end = None if previous_open_ended else row.end_date
