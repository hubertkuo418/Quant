from __future__ import annotations

import pandas as pd

from scripts.generate_smoke_market import generate_smoke_market


def test_smoke_market_is_deterministic_and_valid_ohlcv() -> None:
    first = generate_smoke_market(
        start="2024-01-01", days=20, tickers=["AAA", "BBB"], seed=7
    )
    second = generate_smoke_market(
        start="2024-01-01", days=20, tickers=["AAA", "BBB"], seed=7
    )

    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )
    assert len(first) == 40
    assert first.groupby("ticker")["date"].nunique().eq(20).all()
    assert (first["high"] >= first[["open", "close"]].max(axis=1)).all()
    assert (first["low"] <= first[["open", "close"]].min(axis=1)).all()
    assert (first[["open", "high", "low", "close", "adj_close"]] > 0).all().all()
    assert (first["volume"] > 0).all()
