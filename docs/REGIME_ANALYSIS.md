# Market Regime Analysis

QuantLab classifies each eligible benchmark date along two dimensions:

- Trend: `bull` when the configured lookback return is non-negative, otherwise
  `bear`.
- Volatility: `high_vol` when rolling annualized volatility is at or above its
  historical rolling median, otherwise `low_vol`.

The volatility threshold is shifted by one trading day before its rolling
median is calculated. Therefore, a date's classification never uses that
date's volatility to set its own threshold.

## Run

```powershell
python scripts/analyze_regimes.py --config configs/regime.yaml
```

The default configuration uses SPY, a 60-day trend, 20-day realized
volatility, and a 252-day historical threshold window.

## Outputs

```text
artifacts/regimes/
  daily_regimes.csv
  regime_performance.csv
  manifest.json
```

`regime_performance.csv` reports observations, date range, compounded and
annualized return, annual volatility, Sharpe ratio, maximum drawdown, and win
rate for every observed trend/volatility regime.

Regime analysis is diagnostic rather than a trading signal. The current
implementation does not change portfolio weights based on the regime label.
