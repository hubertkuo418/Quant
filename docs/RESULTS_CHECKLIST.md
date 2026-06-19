# Strategy Studio Release Checklist

Complete this checklist before publishing performance numbers in the README,
resume, strategy catalog, or generated reports.

## Reproducibility

- [ ] Save the exact versioned `StrategySpec` used by every run.
- [ ] Record the run ID, strategy hash, market-data hash, and signal-data hash.
- [ ] Record data provider, download timestamp, universe, and date range.
- [ ] Record Git commit SHA and Python/package versions.
- [ ] Preserve the generated metrics, returns, weights, trades, and run manifest.
- [ ] Confirm the result can be rebuilt with repository scripts only.

## Data Integrity

- [ ] Acknowledge survivorship bias and point-in-time data limitations.
- [ ] State whether prices are adjusted for corporate actions.
- [ ] Confirm features and signals do not use future information.
- [ ] Fit normalizers and learned components on training data only.
- [ ] Keep test-period results out of model and parameter selection.

## Backtest Assumptions

- [ ] State rebalance frequency, holding horizon, and execution lag.
- [ ] State commission, slippage, cash rate, and short borrow assumptions.
- [ ] State liquidity, position, sector, and leverage constraints.
- [ ] Compare strategies and benchmarks over the same common period.
- [ ] Reject zero-lag results from formal optimizer and recommendation outputs.

## Required Results

- [ ] Annual return, volatility, Sharpe, Sortino, and Calmar ratios.
- [ ] Maximum drawdown, turnover, transaction cost, and win rate.
- [ ] VaR, CVaR, Profit Factor, and benchmark-relative metrics.
- [ ] Equity curve, drawdown curve, exposure, and contribution diagnostics.
- [ ] Comparison against simple benchmark strategies.
- [ ] Results across market regimes when the history is long enough.

## Robustness

- [ ] Rerun with doubled costs and one additional day of execution lag.
- [ ] Test nearby parameter values instead of reporting only the best point.
- [ ] Check Top-1/Top-5 contribution share and effective contributors.
- [ ] Check whether one market regime explains nearly all performance.
- [ ] Apply profile constraints before recommending a strategy.
- [ ] Label recommendations as decision support, not guaranteed suitability.

## Product Quality

- [ ] Validate StrategySpec create, edit, save, and reload flows.
- [ ] Validate registry browsing and common-period comparisons.
- [ ] Validate optimizer constraints and Pareto candidate output.
- [ ] Validate recommendation profiles and generated reports.
- [ ] Run the full automated test suite, Ruff, and `git diff --check`.
- [ ] Verify the Streamlit workflow in a real browser.

Publish only results regenerated from immutable run artifacts. Short histories,
unadjusted prices, static universes, and incomplete fundamentals must remain
visible beside the metrics they affect.
