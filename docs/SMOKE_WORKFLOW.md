# Deterministic Smoke Workflow

The smoke workflow verifies runtime contracts when live market data is
temporarily unavailable. It uses deterministic synthetic OHLCV and must never
be presented as evidence of investment performance.

## Generate Synthetic Market Data

```powershell
python scripts/generate_smoke_market.py --days 500 --seed 42
```

The generator writes `data/metadata/smoke_market_manifest.json` with
`synthetic: true` and `purpose: pipeline smoke testing only`.

## Factor And Portfolio Path

```powershell
python scripts/build_features.py --config configs/features.yaml
python scripts/build_alphas.py --config configs/alphas.yaml
python scripts/build_factor_panel.py --config configs/factor_panel.yaml
python scripts/validate_factors.py --config configs/factors.yaml
python scripts/select_factors.py --config configs/factor_selection.yaml
python scripts/build_factor_signals.py --config configs/factor_signals.yaml
python scripts/build_strategy.py --config configs/strategy.yaml
python scripts/run_backtest.py --config configs/backtest.yaml
python scripts/run_benchmarks.py --config configs/benchmarks.yaml
python scripts/run_sensitivity.py --config configs/sensitivity.yaml
python scripts/build_attribution.py --config configs/attribution.yaml
python scripts/analyze_regimes.py --config configs/regime.yaml
```

The default smoke universe has eight symbols. Because the production factor
strategy selects the top ten, it degenerates to equal weight in this smoke run.
That behavior validates plumbing only; it does not validate factor efficacy.

## Model Path

```powershell
python scripts/build_dataset.py --config configs/smoke/dataset.yaml
python scripts/run_baselines.py --config configs/smoke/baselines.yaml
python scripts/train_recurrent.py --config configs/smoke/recurrent.yaml
python scripts/train_transformer.py --config configs/smoke/transformer.yaml
python scripts/build_prediction_signals.py `
  --config configs/smoke/prediction_signals.yaml
python scripts/build_strategy.py --config configs/smoke/model_strategy.yaml
python scripts/run_backtest.py --config configs/smoke/model_backtest.yaml
```

Smoke model configs use reduced capacity and two training epochs. Their metrics
only prove that training, prediction, signal conversion, and backtesting run
end to end.

## Live-Data Boundary

The smoke workflow does not replace a frozen Yahoo or licensed market dataset.
If Yahoo returns `YFRateLimitError`, wait for the provider limit to reset or use
an approved alternate provider before producing publishable results.
