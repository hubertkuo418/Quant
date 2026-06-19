# Return Attribution And Concentration

QuantLab attributes daily gross asset returns by ticker:

```text
return_contribution = portfolio_weight * asset_return
```

This intentionally excludes transaction costs, cash interest, and borrow fees.
Those items remain portfolio-level diagnostics because assigning them to
individual stocks would require additional allocation assumptions.

## Run

Run the backtest first so that holdings exist, then build attribution:

```powershell
python scripts/run_backtest.py --config configs/backtest.yaml
python scripts/build_attribution.py --config configs/attribution.yaml
```

## Outputs

```text
artifacts/attribution/
  daily_attribution.parquet
  ticker_summary.csv
  metrics.json
```

Ticker summaries include total contribution, absolute contribution, average
weight, days held, contribution share, and rank. Concentration metrics include:

- Top-1 absolute contribution share.
- Top-5 absolute contribution share.
- Effective contributors, computed as the inverse Herfindahl index of absolute
  contribution shares.

A high Sharpe ratio with very few effective contributors should be treated as
concentrated evidence, not broad model validation.
