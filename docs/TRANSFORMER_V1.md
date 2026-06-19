# Transformer

The Transformer module supports both the original single-horizon experiment and
a multi-horizon head for all configured dataset horizons.

## Architecture

```text
[batch, 60, 13]
  -> linear input projection
  -> learned positional embedding
  -> pre-norm Transformer encoder
  -> final time-step representation
  -> layer normalization + linear head
  -> [batch, selected_horizons]
```

Default configuration:

- Model dimension: 64.
- Attention heads: 4.
- Encoder layers: 2.
- Feed-forward dimension: 128.
- Dropout: 0.2.

## Training

- Huber loss for robustness to extreme realized returns.
- Optional same-date Pearson correlation loss for cross-sectional ranking.
- AdamW optimizer with weight decay.
- Gradient norm clipping.
- Validation early stopping with best-state restoration.
- Fixed random seed for reproducibility.

The test set is evaluated only after the best validation checkpoint is
selected. Saved artifacts include the checkpoint, test predictions, training
history, and prediction metrics.

`training.correlation_loss_weight` controls the ranking-aware term:

```text
total_loss = Huber(prediction, target) + lambda * (1 - cross_sectional_correlation)
```

When `lambda > 0`, training and validation use date-grouped batches containing
all available stocks for one date. This prevents correlation from being
incorrectly computed across unrelated dates. Single-stock or zero-variance
cross-sections safely fall back to Huber loss. Set the value to `0.0` to retain
the original shuffled mini-batch training behavior.

Set `target_horizon` to an integer for a single-horizon run. Set it to `null`
to train one shared Transformer head across all dataset horizons, for example
5/20/60 trading-day returns. Multi-horizon runs save one prediction row per
`(date, ticker, horizon)` and include `metrics_by_horizon` in `metrics.json`.

## Correctness Checks

Automated tests verify:

- Model output shape.
- Ability to overfit a tiny deterministic batch.
- End-to-end training and prediction on synthetic chronological data.
- Multi-horizon output and per-horizon metric persistence.
- Date-grouped cross-sections and correlation-loss numerical stability.
- Checkpoint and training-history creation.

The tiny-batch test is especially important: a model that cannot memorize a
small batch usually has a broken target, optimizer, tensor shape, or gradient
path and should not be trusted on the full dataset.
