# Recurrent Models

QuantLab includes a shared recurrent sequence baseline for:

```text
RNN
LSTM
GRU
```

The configured `model_type` selects the recurrent encoder. The model consumes
the same `[batch, sequence_length, feature_dim]` dataset as the Transformer and
predicts all configured horizons jointly.

## Workflow

```powershell
python scripts/train_recurrent.py --config configs/recurrent.yaml
```

Outputs:

```text
artifacts/recurrent/
  best_model.pt
  metrics.csv
  test_predictions.parquet
  training_history.csv
  run.json
```

This gives RQ3 a fair comparison path between MLP, recurrent models, and the
Transformer on the same purged chronological dataset.
