# Live Data Experiment

## Frozen scope

- Provider: Nasdaq historical quote endpoint
- Requested period: 2024-12-18 through 2026-06-17
- Universe: 30 liquid US equities plus SPY as benchmark
- Rows: 11,593
- Market dates: 374
- Split cutoffs: train 2025-10-27, validation 2026-02-23
- Sequence length: 60 market days
- Forecast horizons: 5, 20, and 60 market days
- LSTM: one 32-unit layer, at most 8 epochs, validation early stopping
- Transformer: one 32-dimensional layer, at most 4 epochs, validation early stopping

The split cutoffs are chosen on the observed market calendar. The 60-day target
must realize before each split boundary, so boundary observations are purged.
Model capacity is deliberately small for the limited live-data sample; the
goal is a fair out-of-sample comparison rather than maximizing parameter count.

## Data limitations

Nasdaq supplies unadjusted OHLCV through this endpoint. The canonical panel
therefore records close as `adj_close`, and the metadata manifest states this
policy. Splits and dividends may create artificial returns. Published results
must retain this limitation.

Point-in-time fundamental and news ingestion is implemented but no licensed
source is configured. The live model experiment excludes P/E, P/B, and news
columns rather than imputing unavailable data. It currently uses nine trailing
technical and price-volume features.

The universe is a present-day static list, so the experiment remains exposed
to survivorship bias. Results are research evidence, not a production trading
claim.
