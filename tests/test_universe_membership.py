from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.data.universe import (
    filter_market_to_membership,
    load_universe_membership,
    membership_tickers,
)


def write_membership(tmp_path: Path, rows: list[dict[str, object]]) -> Path:
    path = tmp_path / "membership.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_point_in_time_membership_filters_each_market_date(tmp_path: Path) -> None:
    membership = load_universe_membership(
        write_membership(
            tmp_path,
            [
                {
                    "ticker": "AAA",
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                },
                {
                    "ticker": "BBB",
                    "start_date": "2024-01-03",
                    "end_date": "",
                },
            ],
        )
    )
    market = pd.DataFrame(
        [
            {"date": date, "ticker": ticker, "adj_close": 100.0}
            for date in pd.date_range("2024-01-01", periods=4)
            for ticker in ("AAA", "BBB", "SPY")
        ]
    )

    filtered = filter_market_to_membership(market, membership, ("SPY",))

    assert membership_tickers(membership) == ("AAA", "BBB")
    assert filtered.loc[filtered["ticker"] == "AAA", "date"].max() == pd.Timestamp(
        "2024-01-02"
    )
    assert filtered.loc[filtered["ticker"] == "BBB", "date"].min() == pd.Timestamp(
        "2024-01-03"
    )
    assert len(filtered.loc[filtered["ticker"] == "SPY"]) == 4


@pytest.mark.parametrize(
    "rows",
    [
        [
            {"ticker": "AAA", "start_date": "2024-01-01", "end_date": ""},
            {
                "ticker": "AAA",
                "start_date": "2024-02-01",
                "end_date": "2024-03-01",
            },
        ],
        [
            {
                "ticker": "AAA",
                "start_date": "2024-01-01",
                "end_date": "2024-02-01",
            },
            {
                "ticker": "AAA",
                "start_date": "2024-01-15",
                "end_date": "2024-03-01",
            },
        ],
    ],
)
def test_membership_rejects_overlapping_intervals(
    tmp_path: Path,
    rows: list[dict[str, object]],
) -> None:
    with pytest.raises(ValueError, match="overlap"):
        load_universe_membership(write_membership(tmp_path, rows))
