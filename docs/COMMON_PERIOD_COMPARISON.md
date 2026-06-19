# Common-Period Portfolio Comparison

Portfolio metrics are not directly comparable when equity curves cover
different dates. QuantLab aligns every configured curve to the intersection of
their available periods and recomputes the full metric set.

## Method

1. Find the latest start date and earliest end date across all curves.
2. Keep only rows inside that common interval.
3. Set the first common-period return, turnover, and cost to zero.
4. Rebase NAV to 1.0 and compound returns from that point.
5. Recompute return, risk, drawdown, turnover, cost, and tail metrics.

The first-row reset prevents a return earned before the common interval from
leaking into the comparison.

## Run

```powershell
python scripts/compare_equity_curves.py --config configs/equity_comparison.yaml
```

Outputs:

```text
artifacts/comparisons/common_period.csv
artifacts/comparisons/manifest.json
artifacts/comparisons/aligned_curves/<portfolio>.csv
```

The standard config expects factor, model, and SPY equity curves. The smoke
config at `configs/smoke/equity_comparison.yaml` uses explicitly synthetic
artifacts and must not be cited as market evidence.
