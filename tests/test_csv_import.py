from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.data.csv_import import (
    MarketCsvImportConfig,
    MarketCsvImporter,
)


def write_prices(path: Path, *, include_symbol: bool = False) -> None:
    frame = pd.DataFrame(
        {
            "Date": ["2024-01-02", "2024-01-03"],
            "Open": [100, 101],
            "High": [102, 103],
            "Low": [99, 100],
            "Close": [101, 102],
            "Adj Close": [100.5, 101.5],
            "Volume": [1000, 1200],
        }
    )
    if include_symbol:
        frame["Symbol"] = ["aaa", "aaa"]
    frame.to_csv(path, index=False)


def make_config(
    tmp_path: Path, input_path: Path, **kwargs: object
) -> MarketCsvImportConfig:
    return MarketCsvImportConfig(
        input_path=input_path,
        output_path=tmp_path / "processed" / "market.parquet",
        metadata_path=tmp_path / "metadata" / "manifest.json",
        **kwargs,
    )


def test_imports_directory_files_and_records_hashes(tmp_path: Path) -> None:
    source = tmp_path / "csv"
    source.mkdir()
    write_prices(source / "aaa.csv")
    write_prices(source / "BBB.csv")
    config = make_config(tmp_path, source)

    panel = MarketCsvImporter(config).run()

    assert panel.shape == (4, 8)
    assert panel["ticker"].unique().tolist() == ["AAA", "BBB"]
    assert config.output_path.exists()
    manifest = json.loads(config.metadata_path.read_text(encoding="utf-8"))
    assert manifest["synthetic"] is False
    assert len(manifest["sources"]) == 2
    assert all(len(item["sha256"]) == 64 for item in manifest["sources"])


def test_imports_long_file_aliases_and_filters_end_exclusive(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    write_prices(source, include_symbol=True)
    config = make_config(tmp_path, source, end_date="2024-01-03")

    panel = MarketCsvImporter(config).run()

    assert len(panel) == 1
    assert panel.iloc[0]["ticker"] == "AAA"
    assert panel.iloc[0]["adj_close"] == 100.5


def test_single_file_without_ticker_requires_default(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    write_prices(source)

    with pytest.raises(ValueError, match="default_ticker"):
        MarketCsvImporter(make_config(tmp_path, source)).run()


def test_import_rejects_invalid_numeric_values(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    write_prices(source, include_symbol=True)
    frame = pd.read_csv(source)
    frame["Close"] = frame["Close"].astype("object")
    frame.loc[0, "Close"] = "bad"
    frame.to_csv(source, index=False)

    with pytest.raises(ValueError, match="Invalid required values"):
        MarketCsvImporter(make_config(tmp_path, source)).run()
