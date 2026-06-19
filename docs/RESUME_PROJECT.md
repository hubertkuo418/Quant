# Resume Project Brief

## Project Title

**QuantLab Strategy Studio: Custom Quant Strategy Design, Backtesting, Comparison, and Optimization**

## One-Line Summary

Built a versioned quantitative strategy platform that lets users compose
factor, technical, and model signals; apply portfolio and risk constraints;
run cost-aware backtests; compare strategies on common periods; and search for
constraint-aware Pareto candidates.

## Resume Bullets

- Designed a declarative `StrategySpec` contract for configurable universes,
  multi-signal composition, Top-K/long-short construction, weighting,
  rebalancing, risk limits, costs, execution lag, and benchmarks.
- Built a reproducible Strategy Run Registry that stores data/signal/spec
  hashes, package and Git versions, target weights, trades, equity curves, and
  performance metrics for every strategy version.
- Implemented common-period multi-strategy comparison and grid/random parameter
  search with user constraints and multi-objective Pareto-frontier selection.
- Added personalized conservative, balanced, and aggressive candidate ranking
  using return, Sharpe, drawdown, turnover, and execution-lag robustness.
- Developed a shared portfolio and backtest engine covering commission,
  slippage, liquidity, execution lag, cash, borrow costs, position/sector caps,
  attribution, regime analysis, and sensitivity testing.
- Integrated technical/factor signals, Ridge/XGBoost/LightGBM/MLP, recurrent
  models, and a multi-horizon Transformer as interchangeable signal components.
- Delivered YAML/CLI workflows, DuckDB artifact cataloging, automated tests,
  and a Streamlit interface for strategy design, execution, comparison,
  optimization, and run management.

## Interview Talking Points

1. **Platform contracts:** strategy definitions are data, while portfolio and
   backtest behavior stays centralized and tested.
2. **Fair comparison:** all candidates are rebased and recomputed on the same
   common period before constraints or objectives are applied.
3. **Reproducibility:** every run records exact specs and SHA-256 fingerprints
   of both market and signal inputs.
4. **Robust selection:** recommendations consider lag sensitivity, drawdown,
   turnover, and Pareto tradeoffs instead of maximizing one backtest metric.
5. **Extensibility:** rules, factors, custom composite scores, and model
   predictions all enter the same signal-to-portfolio contract.
6. **Honest limitations:** present-day universe membership, short live-data
   history, unadjusted Nasdaq close, and simplified execution remain explicit.

## Metrics Rule

Use performance numbers only when the exact `run_id`, data hash, StrategySpec,
common period, costs, and limitations are linked. Do not present optimizer
ranking as a promise of future performance or personalized financial advice.
