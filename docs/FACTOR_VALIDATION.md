# Factor Validation

QuantLab separates feature creation from factor validation:

- `features/` creates point-in-time columns.
- `factors/` identifies which columns are research factors and evaluates them.

## Workflow

```powershell
python scripts/validate_factors.py --config configs/factors.yaml
```

The validation pipeline reads a feature panel, creates a forward target when
needed, infers factor families, and writes:

```text
artifacts/factors/
  coverage.csv
  daily_ic.csv
  ic_summary.csv
  quantile_returns.csv
  quantile_summary.csv
  manifest.json
```

## Metrics

- **Coverage:** percent of non-null observations per factor.
- **IC:** mean daily cross-sectional Pearson correlation.
- **Rank IC:** mean daily cross-sectional Spearman correlation.
- **ICIR:** mean IC divided by IC standard deviation.
- **Positive IC Rate:** fraction of evaluated dates with positive IC.
- **Quantile Return:** average forward return by factor bucket.

Factor directions are applied before evaluation. For example, lower P/E and
lower volatility are treated as better signals by default.

## Role in QuantLab

This module powers the future Factor Explorer GUI and provides the statistical
filter before signals enter strategy construction or model training.

## Factor Selection

After validation, select stable candidate factors:

```powershell
python scripts/select_factors.py --config configs/factor_selection.yaml
```

Outputs:

```text
artifacts/factors/selected_factors.csv
artifacts/factors/selected_factors.json
```

Selection filters on coverage, evaluated periods, absolute Rank IC, and positive
IC rate, then ranks candidates by a simple stability score.

## Factor Signals

Selected factors can be blended into a strategy-ready `factor_score` using
their signed Rank IC:

```powershell
python scripts/build_factor_signals.py --config configs/factor_signals.yaml
```

The signal builder cross-sectionally z-scores each selected factor by date,
weights each factor by signed `mean_rank_ic`, preserves configured passthrough
columns such as `volatility_20d`, and writes:

```text
artifacts/factors/factor_signals.parquet
artifacts/factors/factor_signals.json
```
