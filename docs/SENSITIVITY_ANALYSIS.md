# Backtest Sensitivity Analysis

Sensitivity analysis reruns one fixed set of target weights under alternative
execution and financing assumptions. Market data and strategy signals remain
unchanged, so differences are attributable to the configured stress scenario.

## Default Scenarios

- `baseline`: settings from `configs/backtest.yaml`.
- `double_costs`: 10 bps commission plus 10 bps slippage.
- `execution_lag_1d`: target weights applied one trading day later.
- `execution_lag_2d`: target weights applied two trading days later.
- `high_borrow_cost`: 5% annualized borrow fee on short exposure.
- `combined_stress`: doubled costs, one-day lag, and 5% borrow fee.

Only a whitelist of execution, liquidity, cash, and borrow assumptions can be
overridden. Scenario names are validated before being used as output paths.

## Run

```powershell
python scripts/run_sensitivity.py --config configs/sensitivity.yaml
```

## Outputs

```text
artifacts/sensitivity/
  comparison.csv
  manifest.json
  baseline/
  double_costs/
  execution_lag_1d/
  execution_lag_2d/
  high_borrow_cost/
  combined_stress/
```

Each scenario directory contains the same detailed artifacts as a normal
backtest. `comparison.csv` combines assumptions, ending NAV, and performance
metrics for direct inspection in the Streamlit Dashboard.

Sensitivity analysis should be used to disclose fragility. It should not be
used to select whichever cost or lag assumption produces the highest Sharpe.
