# Unified OOS Robustness

The robustness evaluator applies configured execution and portfolio stresses to
the same frozen strategy and reruns each through the same Walk-forward calendar.
It prevents a scenario with fewer invested days from receiving an unfairly
annualized result by filling all pre-investment test dates as explicit cash
days.

## Scenarios

- baseline assumptions;
- doubled commission and slippage;
- one additional execution-lag day;
- Top-K minus two and plus two;
- monthly instead of weekly rebalancing.

## Outputs

`scenario_summary.csv` includes aggregate Sharpe, drawdown, turnover, worst
fold Sharpe, fold Sharpe dispersion, positive-fold rate, and constraint status.
`aggregate.json` records pass rate, median and worst Sharpe, dispersion, and
worst drawdown. Each scenario retains its own StrategySpec and immutable fold
runs.

## Current Boundary

The formal factor-top10 run contains six scenarios over the same 60 OOS trading
days. All pass the current permissive constraints, but doubled costs and added
lag each contain one negative-Sharpe fold. The short evaluation history remains
the dominant limitation.
