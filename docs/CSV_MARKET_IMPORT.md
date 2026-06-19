# CSV Market Data Import

Use the CSV importer when the live Nasdaq endpoint is unavailable or
when market data was obtained from another licensed source.

## Accepted layouts

The importer accepts either:

1. One long-form CSV containing a `ticker` or `symbol` column.
2. A directory of one file per ticker named `<TICKER>.csv`.

Required fields are `date`, `open`, `high`, `low`, `close`, and `volume`.
Headers are case-insensitive. `Adj Close`, `adjusted_close`, `symbol`,
`datetime`, and `timestamp` are recognized aliases. If adjusted close is not
provided, the importer uses close.

For a single-symbol file without a ticker column, set `default_ticker` in
`configs/csv_import.yaml`.

## Run

Place input files under `data/import/market_csv`, review the config, then run:

```powershell
python scripts/import_market_csv.py --config configs/csv_import.yaml
```

The output uses the same canonical schema as the live data pipeline:

```text
date, ticker, open, high, low, close, adj_close, volume
```

By default it writes `data/processed/market_panel.parquet`, matching the live
downloader and every downstream feature job.

The configured end date is exclusive, matching the live downloader. The
import fails on missing values, invalid prices, fractional volume, duplicate
`(date, ticker)` keys, or an empty date range.

## Data lineage

The manifest records the exact source path, byte size, SHA-256 hash, row count,
ticker list, and imported date range. It also marks the dataset as
`synthetic: false`; this describes provenance, not a claim about the vendor or
license. Keep source files private when their license prohibits redistribution.
