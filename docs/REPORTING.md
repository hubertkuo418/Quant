# Reporting

The reporting layer aggregates existing artifacts into dashboard-ready tables.

## Workflow

```powershell
python scripts/build_report.py --config configs/reporting.yaml
```

Outputs:

```text
artifacts/reports/model_comparison.csv
artifacts/reports/portfolio_comparison.csv
artifacts/reports/trading_diagnostics.csv
```

These files are read by the Streamlit dashboard. Reporting does not retrain
models or rerun backtests; it only combines already-produced metrics.

Model comparison can include baseline tabular models, recurrent sequence
models, and Transformer metrics when their artifact files exist.

Trading diagnostics summarize trade count, total traded value, total cost,
average exposure, average active positions, and sector concentration when
`sector_exposure.csv` exists.
