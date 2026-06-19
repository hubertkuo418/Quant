# Baseline Experiments

Week 4 establishes the minimum forecasting standards that the Transformer must
beat on the same purged chronological dataset.

## Models

| Model | Input | Purpose |
|---|---|---|
| Historical mean | Train target mean | Constant no-skill return forecast |
| Momentum | Latest normalized 20-day return | Calibrated univariate momentum |
| Ridge | Latest feature vector | Stable linear cross-sectional baseline |
| Random Forest | Latest feature vector | Bagged nonlinear tree baseline |
| XGBoost | Latest feature vector | Nonlinear tabular baseline |
| LightGBM | Latest feature vector | Fast gradient-boosted tree baseline |
| MLP | Flattened 60-day feature sequence | Simple sequence-aware neural baseline |

Ridge, momentum, and XGBoost train a separate regressor for each horizon. The
MLP shares hidden layers and predicts all horizons jointly.

## Selection and Evaluation

- Training rows fit model parameters.
- Validation rows support XGBoost monitoring and MLP early stopping.
- Test rows are evaluated only after fitting is complete.
- Every test prediction is saved with model, date, ticker, horizon, and target.

Reported metrics:

- MAE and RMSE.
- Mean daily Pearson information coefficient.
- Mean daily Spearman rank information coefficient.
- Directional accuracy.

IC values are `NaN` for constant cross-sectional forecasts such as the
historical mean because correlation is mathematically undefined.

## Artifacts

Running `scripts/run_baselines.py` creates:

```text
artifacts/baselines/
  metrics.csv
  test_predictions.parquet
  run.json
```

These test predictions will later feed the exact same Week 7-9 portfolio rules
used for Transformer forecasts.
