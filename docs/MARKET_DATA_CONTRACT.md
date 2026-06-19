# Market Data Trust Contract

QuantLab separates data plumbing capability from the quality of the currently
connected source. A provider must disclose its corporate-action adjustment
status, and a universe can optionally be defined by point-in-time membership
intervals.

## Price Adjustment Guard

Every provider reports one of these manifest statuses:

- `provider_adjusted_ohlcv`: OHLC values are provider-adjusted;
- `provider_adjusted_close`: adjusted close is provider-reported;
- `unadjusted_close_proxy`: close is copied to `adj_close` without adjustment;
- `unknown`: the adapter cannot establish adjustment provenance.

Set `require_adjusted_prices: true` in `configs/data.yaml` for a formal workflow.
The pipeline then rejects providers that cannot supply adjusted prices before
any canonical panel is replaced. The current Nasdaq adapter reports
`unadjusted_close_proxy`, so this guard deliberately rejects it.

## Point-in-Time Universe

Set `universe.membership_path` to a CSV with inclusive membership intervals:

```csv
ticker,start_date,end_date
AAPL,2020-01-01,
REMOVED,2020-01-01,2022-12-31
```

Blank `end_date` means the membership remains active. Intervals for one ticker
must not overlap. The pipeline downloads the union of all member tickers, then
keeps a row only when that ticker was active on the row's date.
`always_include_tickers` keeps benchmarks such as SPY outside the membership
filter. See `examples/universe_membership.csv` for a versioned template.

## Manifest Evidence

Each download manifest records:

- `price_adjustment_status` and whether adjusted prices were required;
- `universe_membership_policy` as static or point-in-time;
- requested and successful tickers, failures, dates, and row count;
- the complete resolved configuration.

These contracts remove silent ambiguity, but they do not create historical
membership or corporate-action data. A trustworthy source still needs to be
connected before survivorship-bias and adjustment limitations can be retired.
