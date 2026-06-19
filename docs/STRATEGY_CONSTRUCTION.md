# Strategy Construction

The strategy layer converts a dated signal into target portfolio weights. It
does not simulate cash, transaction costs, or daily NAV; those belong to the
backtesting layer.

`excluded_tickers` removes benchmark or non-tradable instruments before
ranking. The default factor and model configs exclude SPY while retaining it
in the market panel for benchmark returns.

## Supported Strategies

```text
long_only_top_k
long_short_quantile
```

## Supported Weighting

```text
equal
score
inverse_volatility
risk_parity
```

For long-short portfolios, the current convention is 50% gross long and 50%
gross short, so net exposure is zero and gross exposure is one.

`inverse_volatility` and `risk_parity` use `volatility_column` from the signal
panel and allocate more weight to lower-volatility names within each long or
short sleeve. Volatility values must be positive.

## Rebalancing

The module supports daily weights or any pandas period frequency such as
`W-FRI` or `M`. For non-daily frequencies, the rebalance date is the last
available trading date in each period.

## Risk Controls

Optional sector controls:

```text
sector_column
volatility_column
max_sector_weight
max_position_weight
```

When configured, target weights preserve a `sector` column and cap each
sector's gross exposure. Any weight removed by the cap remains unallocated
rather than being redistributed past the cap.

`max_position_weight` similarly caps each single-name absolute weight.

## Workflow

```powershell
python scripts/build_strategy.py --config configs/strategy.yaml
```

Model predictions can be converted into a strategy-ready signal panel first:

```powershell
python scripts/build_prediction_signals.py --config configs/prediction_signals.yaml
```

Then set `signal_path` to `artifacts/strategies/model_signals.parquet` and
`score_column` to `model_score` in `configs/strategy.yaml`.

Outputs:

```text
artifacts/strategies/
  target_weights.parquet
  manifest.json
```

These target weights are the contract consumed by the future backtesting engine.
