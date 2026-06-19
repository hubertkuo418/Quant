# QuantLab Workflow

This is the current end-to-end research path:

```powershell
python scripts/download_market_data.py --config configs/data.yaml
# Alternative: python scripts/import_market_csv.py --config configs/csv_import.yaml
python scripts/analyze_market_quality.py --config configs/data_quality.yaml
python scripts/build_features.py --config configs/features.yaml
python scripts/build_alphas.py --config configs/alphas.yaml
python scripts/build_factor_panel.py --config configs/factor_panel.yaml
python scripts/build_catalog.py --config configs/catalog.yaml
python scripts/validate_factors.py --config configs/factors.yaml
python scripts/select_factors.py --config configs/factor_selection.yaml
python scripts/build_factor_signals.py --config configs/factor_signals.yaml
python scripts/build_strategy.py --config configs/strategy.yaml
python scripts/run_backtest.py --config configs/backtest.yaml
python scripts/run_benchmarks.py --config configs/benchmarks.yaml
python scripts/analyze_regimes.py --config configs/regime.yaml
python scripts/run_sensitivity.py --config configs/sensitivity.yaml
python scripts/build_attribution.py --config configs/attribution.yaml
python scripts/compare_equity_curves.py --config configs/equity_comparison.yaml
python scripts/build_report.py --config configs/reporting.yaml
python scripts/build_dataset.py --config configs/dataset.yaml
python scripts/run_baselines.py --config configs/baselines.yaml
python scripts/train_recurrent.py --config configs/recurrent.yaml
python scripts/train_transformer.py --config configs/transformer.yaml
python scripts/build_prediction_signals.py --config configs/prediction_signals.yaml
python scripts/build_strategy.py --config configs/model_strategy.yaml
python scripts/run_backtest.py --config configs/model_backtest.yaml
python scripts/compare_equity_curves.py --config configs/equity_comparison.yaml
```

The first five commands now form a full factor-research vertical slice:

```text
Data -> Features / Alphas -> Factor Validation -> Factor Signals -> Target Weights -> Backtest -> Benchmarks
```

The last three commands form the model-research slice:

```text
Features -> Sequence Dataset -> Baselines / Transformer -> Prediction Signals
  -> Model Target Weights -> Model Backtest -> Common-Period Comparison
```

## Current Contracts

### Market Data Input

The live downloader and local CSV importer both produce:

```text
data/processed/market_panel.parquet
```

CSV imports additionally write a source-hash manifest configured in
`configs/csv_import.yaml`. See [CSV_MARKET_IMPORT.md](CSV_MARKET_IMPORT.md).

Before feature generation, market diagnostics are written under
`artifacts/data_quality/`. See [DATA_QUALITY.md](DATA_QUALITY.md).

### Feature Panel

One row per `(date, ticker)` with point-in-time factor columns and optional
model targets.

### Factor Validation Outputs

```text
artifacts/factors/coverage.csv
artifacts/factors/daily_ic.csv
artifacts/factors/ic_summary.csv
artifacts/factors/quantile_summary.csv
artifacts/factors/selected_factors.csv
artifacts/factors/factor_signals.parquet
```

### Strategy Outputs

```text
artifacts/strategies/target_weights.parquet
```

Target weights use:

```text
date, ticker, weight, side, score
```

Prediction conversion is controlled by `configs/prediction_signals.yaml` and
writes a manifest containing the selected model/horizon, source hash, row
count, and signal date range.

### Backtest Outputs

```text
artifacts/backtests/equity_curve.csv
artifacts/backtests/holdings.parquet
artifacts/backtests/exposure.csv
artifacts/backtests/sector_exposure.csv
artifacts/backtests/metrics.json
```

### Regime Analysis Outputs

```text
artifacts/regimes/daily_regimes.csv
artifacts/regimes/regime_performance.csv
artifacts/regimes/manifest.json
```

### Sensitivity Outputs

```text
artifacts/sensitivity/comparison.csv
artifacts/sensitivity/manifest.json
artifacts/sensitivity/<scenario>/metrics.json
```

### Attribution Outputs

```text
artifacts/attribution/daily_attribution.parquet
artifacts/attribution/ticker_summary.csv
artifacts/attribution/metrics.json
```

### Common-Period Comparison Outputs

```text
artifacts/comparisons/common_period.csv
artifacts/comparisons/manifest.json
artifacts/comparisons/aligned_curves/
```

The Streamlit GUI includes a workflow tab that can trigger known pipeline
scripts and records run metadata under:

```text
artifacts/workflow_runs/
```

## Next Platform Steps

1. Expand the 20-signal formulaic alpha registry toward full Alpha101 coverage.
2. Add more model-to-portfolio comparison recipes.
3. Add broker-specific margin and tiered borrow-rate assumptions.
4. Add walk-forward retraining recipes for model portfolio experiments.
