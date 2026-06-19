# Strategy Studio Guide

## Create a strategy

Copy an existing file under `strategies/` and change its name, version, signal,
portfolio, risk, or execution settings. A StrategySpec is declarative: it does
not contain backtest code and cannot silently replace shared engine behavior.

```powershell
python scripts/run_studio_strategy.py --spec strategies/factor_top10.yaml
```

The run directory contains:

```text
manifest.json
strategy.yaml
strategy_manifest.json
target_weights.parquet
backtest/equity_curve.csv
backtest/metrics.json
backtest/holdings.parquet
backtest/trade_log.parquet
```

The manifest records hashes of the market and signal inputs. Changing any
StrategySpec field changes its stable spec hash.

## Compare strategies

Choose at least two run IDs:

```powershell
python scripts/compare_studio_runs.py RUN_A RUN_B
```

Returns and metrics are recomputed on the latest common start and earliest
common end. This prevents a strategy with a favorable but shorter period from
winning merely because its dates differ.

## Optimize a strategy

Edit `configs/studio_optimizer.yaml`, then run:

```powershell
python scripts/optimize_studio_strategy.py --config configs/studio_optimizer.yaml
```

Supported V1 methods are exhaustive grid search and reproducible random search.
Parameter names use dotted StrategySpec paths such as `portfolio.top_k` or
`execution.execution_lag_days`.

The optimizer:

1. creates a complete versioned run for every candidate;
2. aligns every candidate to a common period;
3. applies minimum and maximum metric constraints;
4. ranks by the primary objective;
5. marks multi-objective Pareto-efficient candidates.

Use realistic execution lags. Lag-zero results may be useful diagnostics but
should not be treated as deployable recommendations.

## Personalize candidate ranking

Edit `configs/studio_profile.yaml` to select conservative, balanced, or
aggressive preferences and set maximum drawdown, maximum turnover, and minimum
annual return constraints. Then run:

```powershell
python scripts/recommend_studio_strategies.py `
  --profile configs/studio_profile.yaml
```

The recommendation score combines return, Sharpe, drawdown, turnover, and
execution robustness. Execution robustness uses the worst Sharpe and dispersion
across lag assumptions for otherwise matching strategy parameters.

## GUI

Run `streamlit run app/streamlit_app.py` and open the **Strategy Studio** tab to
edit a spec, run it, browse the Registry, compare selected runs, or launch the
configured optimizer.
