# QuantLab Strategy Studio V1.1 Plan

V1.1 focuses on making strategy creation accessible without weakening the
research and reproducibility contracts established in V1.

| Milestone | Deliverable | Status |
|---|---|---|
| 1 | Chinese no-code strategy wizard with advanced YAML fallback | Complete |
| 2 | Strategy duplicate, version history, diff, archive, and delete flows | Complete |
| 3 | Frozen-strategy walk-forward and rolling OOS orchestration | Complete |
| 4 | Cost, lag, liquidity, and nearby-parameter robustness report | Next |
| 5 | Chinese investor-needs questionnaire and constraint translation | Planned |
| 6 | Interactive multi-objective comparison and recommendation rationale | Planned |
| 7 | Corporate-action-adjusted data and historical-universe adapters | Planned |

## Milestone 1 Acceptance

- The visual wizard covers identity, signals, universe, portfolio, risk, and
  execution assumptions.
- Multi-signal components can be added and removed in a table editor.
- Submissions produce the same validated, hash-stable `StrategySpec` as YAML.
- Users can save a strategy or save and immediately backtest it.
- Advanced YAML mode remains available.
- GUI/Studio regression tests, Ruff, and Streamlit AppTest pass.

## Milestones 2-3 Acceptance

- Strategy updates automatically snapshot the previous hash-stable YAML.
- Duplicate, diff, archive, soft-delete, and restore flows are available.
- Rolling and anchored windows support a configurable purge gap.
- Test windows cannot overlap, and evaluation starts no earlier than strategy
  formation unless explicitly configured.
- Each fold is an immutable Strategy Studio run and aggregate OOS returns are
  stitched into a reproducible equity curve.
- Frozen-strategy evaluation is labeled separately from adaptive per-fold
  model retraining, which remains future work.

## Delivery Order

The next implementation slice is the unified robustness report. Profile
recommendations should then consume OOS and robustness gates before ranking
candidates.
