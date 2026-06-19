# Walk-Forward Strategy Evaluation

QuantLab's current Walk-forward mode measures a frozen StrategySpec across
rolling out-of-sample windows. It is designed for execution and portfolio
robustness, not as evidence that a model was retrained inside every fold.

## Contract

- Training and test windows use shared market/signal trading dates.
- A configurable purge gap separates every training and test window.
- Test windows cannot overlap.
- The first test date defaults to `StrategySpec.universe.start_date`, preventing
  evaluation before the recorded strategy formation date.
- Every fold runs through `StrategyStudioRunner`, including normal costs,
  execution lag, liquidity rules, and immutable manifests.
- Daily fold returns are stitched chronologically into one OOS equity curve.

## Run

```powershell
python scripts/run_studio_walk_forward.py `
  --config configs/studio_walk_forward.yaml
```

Outputs are written under `artifacts/studio/walk_forward/<strategy>`:

- `folds.csv`: train/test dates, run IDs, and per-fold metrics;
- `oos_equity_curve.csv`: stitched non-overlapping OOS returns and NAV;
- `metrics.json`: aggregate OOS metrics;
- `manifest.json`: evaluation mode, config, dates, and strategy hash;
- `runs/`: immutable Strategy Studio artifacts for each fold.

## Current Formal Result

The factor-top10 evaluation produced three complete 20-day OOS folds from
March to June 2026. Aggregate Sharpe is high, but the sample is short and the
second fold dominates performance. This result is useful as a pipeline and
robustness checkpoint, not as long-horizon investment evidence.

Adaptive Walk-forward retraining and per-fold factor selection remain separate
future capabilities and must not be inferred from this frozen-strategy result.
