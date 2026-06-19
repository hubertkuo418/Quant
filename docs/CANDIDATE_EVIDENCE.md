# Candidate-Specific OOS Evidence

QuantLab evaluates each feasible Pareto candidate with its own immutable
`StrategySpec`. This prevents the recommendation layer from assigning one base
strategy's Walk-forward result to every optimized candidate.

## Run

```powershell
python scripts/build_candidate_evidence.py `
  --config configs/studio_candidate_evidence.yaml
```

The evaluator selects feasible Pareto rows from the optimizer output, resolves
each `run_id` to `artifacts/studio/runs/<run_id>/strategy.yaml`, and runs the
configured OOS robustness scenarios. Outputs are stored under:

```text
artifacts/studio/evidence/factor_search/
  evidenced_candidates.csv
  manifest.json
  candidates/<run_id>/
```

The enriched table includes baseline OOS return, Sharpe ratio, drawdown, and
observation count plus aggregate robustness pass rate and worst-case metrics.
Strict investor-profile recommendations read this table instead of the raw
optimizer output.

## Current Evidence Boundary

The current formal run contains four candidates and 60 OOS trading dates per
candidate. All configured scenarios pass the present thresholds, while OOS
Sharpe ratios range from about 3.35 to 4.43. These values are useful as a
pipeline checkpoint, but the sample is too short and the underlying panel is
unadjusted and survivorship-biased. They are not deployment-grade performance
claims.
