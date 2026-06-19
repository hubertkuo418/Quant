# Benchmarks

QuantLab benchmark backtests use the same `BacktestEngine` and metric schema as
model or factor strategies.

## Current Benchmarks

- **Equal-Weight Universe:** equal weight across all available tickers on each
  rebalance date.
- **Buy-and-Hold Ticker:** one initial 100% long position in a configured ticker
  such as `SPY`.
- **Momentum Top-K:** rank by trailing adjusted-close return and hold the top K
  names on each rebalance date.

## Workflow

```powershell
python scripts/run_benchmarks.py --config configs/benchmarks.yaml
```

Outputs:

```text
artifacts/benchmarks/
  comparison.csv
  manifest.json
  equal_weight_universe/
  buy_and_hold_SPY/
```

The `comparison.csv` table is designed to sit next to strategy and model
backtest metrics in the future Streamlit dashboard.
