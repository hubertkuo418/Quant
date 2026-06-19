# Market Data Quality

Run market quality checks after downloading or importing OHLCV and before
building features:

```powershell
python scripts/analyze_market_quality.py --config configs/data_quality.yaml
```

The analyzer validates the canonical market schema and reports, per ticker:

- coverage against the union of observed market dates;
- missing dates;
- zero-volume rate;
- longest run of unchanged adjusted close;
- maximum absolute one-day adjusted return.

Thresholds are configurable. Dates are compared with the panel's observed
calendar rather than a generic weekday calendar, so exchange holidays are not
automatically mislabeled as missing.

Outputs:

```text
artifacts/data_quality/per_ticker.csv
artifacts/data_quality/issues.csv
artifacts/data_quality/summary.json
```

An issue is a diagnostic flag, not an automatic deletion rule. Investigate
corporate actions, trading halts, new listings, and vendor corrections before
excluding data.
