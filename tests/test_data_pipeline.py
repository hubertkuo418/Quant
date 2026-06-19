from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.data.config import DataConfig, UniverseConfig
from equity_transformer.data.pipeline import MarketDataPipeline
from equity_transformer.data.providers import MarketDataProvider, NasdaqProvider
from equity_transformer.data.validation import validate_market_frame


class FakeProvider(MarketDataProvider):
    def download(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [102.0, 103.0],
                "Low": [99.0, 100.0],
                "Close": [101.0, 102.0],
                "Adj Close": [100.5, 101.5],
                "Volume": [1_000, 1_200],
            },
            index=pd.DatetimeIndex(["2024-01-02", "2024-01-03"], name="Date"),
        )


class EmptyThenSuccessProvider(FakeProvider):
    def __init__(self) -> None:
        self.calls = 0

    def download(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
            return pd.DataFrame()
        return super().download(ticker, start, end, interval, auto_adjust)


class SelectiveFailureProvider(FakeProvider):
    def download(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        auto_adjust: bool,
    ) -> pd.DataFrame:
        return pd.DataFrame() if ticker == "BBB" else super().download(
            ticker, start, end, interval, auto_adjust
        )


def make_config(tmp_path: Path) -> DataConfig:
    return DataConfig(
        provider="fake",
        start_date="2024-01-01",
        end_date="2024-02-01",
        interval="1d",
        auto_adjust=False,
        max_retries=0,
        retry_delay_seconds=0,
        raw_dir=tmp_path / "raw",
        processed_path=tmp_path / "processed" / "panel.parquet",
        metadata_dir=tmp_path / "metadata",
        universe=UniverseConfig(name="test", tickers=("AAA", "BBB")),
    )


def test_pipeline_builds_sorted_panel_and_artifacts(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    panel = MarketDataPipeline(config, FakeProvider()).run()

    assert panel.shape == (4, 8)
    assert panel["ticker"].unique().tolist() == ["AAA", "BBB"]
    assert not panel[["date", "ticker"]].duplicated().any()
    assert config.processed_path.exists()
    assert (config.raw_dir / "AAA.parquet").exists()
    assert len(list(config.metadata_dir.glob("market_*.json"))) == 1


def test_pipeline_retries_an_empty_provider_response(tmp_path: Path) -> None:
    base = make_config(tmp_path)
    config = replace(
        base,
        max_retries=1,
        retry_delay_seconds=0,
        universe=UniverseConfig(name="test", tickers=("AAA",)),
    )
    provider = EmptyThenSuccessProvider()

    panel = MarketDataPipeline(config, provider).run()

    assert provider.calls == 2
    assert len(panel) == 2


def test_validation_rejects_invalid_high() -> None:
    frame = MarketDataPipeline._standardize(
        FakeProvider().download("AAA", "", "", "1d", False),
        "AAA",
    )
    frame.loc[0, "high"] = 1.0

    with pytest.raises(ValueError, match="High price"):
        validate_market_frame(frame)


def test_validation_rejects_non_finite_prices() -> None:
    frame = MarketDataPipeline._standardize(
        FakeProvider().download("AAA", "", "", "1d", False),
        "AAA",
    )
    frame.loc[0, "close"] = float("inf")

    with pytest.raises(ValueError, match="Non-finite"):
        validate_market_frame(frame)


def test_pipeline_does_not_replace_panel_below_success_threshold(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)

    with pytest.raises(RuntimeError, match="success rate"):
        MarketDataPipeline(config, SelectiveFailureProvider()).run()

    assert not config.processed_path.exists()


def test_nasdaq_provider_parses_currency_and_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {
        "data": {
            "tradesTable": {
                "rows": [
                    {
                        "date": "06/17/2026",
                        "close": "$295.95",
                        "volume": "42,745,060",
                        "open": "$300.845",
                        "high": "$302.07",
                        "low": "$294.36",
                    }
                ]
            }
        }
    }
    monkeypatch.setattr(
        NasdaqProvider,
        "_request_json",
        staticmethod(lambda _url: payload),
    )

    frame = NasdaqProvider().download("AAPL", "2026-06-01", "2026-06-18", "1d", False)

    assert frame.loc[pd.Timestamp("2026-06-17"), "Close"] == 295.95
    assert frame.loc[pd.Timestamp("2026-06-17"), "Volume"] == 42_745_060
    assert frame["Adj Close"].equals(frame["Close"])


def test_nasdaq_provider_maps_share_class_and_etf_asset_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    urls: list[str] = []

    def capture(url: str) -> dict[str, object]:
        urls.append(url)
        return {"data": {"tradesTable": {"rows": []}}}

    monkeypatch.setattr(NasdaqProvider, "_request_json", staticmethod(capture))
    provider = NasdaqProvider()

    provider.download("BRK-B", "2026-01-01", "2026-02-01", "1d", False)
    provider.download("SPY", "2026-01-01", "2026-02-01", "1d", False)

    assert "/BRK.B/historical" in urls[0]
    assert "assetclass=stocks" in urls[0]
    assert "/SPY/historical" in urls[1]
    assert "assetclass=etf" in urls[1]


def test_pipeline_rejects_unadjusted_provider_when_required(tmp_path: Path) -> None:
    config = replace(make_config(tmp_path), require_adjusted_prices=True)

    with pytest.raises(ValueError, match="requires corporate-action-adjusted"):
        MarketDataPipeline(config, NasdaqProvider())


def test_pipeline_applies_point_in_time_universe_and_records_policy(
    tmp_path: Path,
) -> None:
    membership_path = tmp_path / "membership.csv"
    pd.DataFrame(
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
        ]
    ).to_csv(membership_path, index=False)
    config = replace(
        make_config(tmp_path),
        universe=UniverseConfig(
            name="historical-test",
            tickers=(),
            membership_path=membership_path,
            always_include_tickers=("SPY",),
        ),
    )

    panel = MarketDataPipeline(config, FakeProvider()).run()

    assert panel.groupby("ticker").size().to_dict() == {
        "AAA": 1,
        "BBB": 1,
        "SPY": 2,
    }
    manifest_path = next(config.metadata_dir.glob("market_*.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["universe_membership_policy"] == "point_in_time_intervals"
    assert manifest["price_adjustment_status"] == "unknown"
