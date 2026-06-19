# QuantLab Strategy Studio V1.1 Plan

V1.1 focuses on making strategy creation accessible without weakening the
research and reproducibility contracts established in V1.

| Milestone | Deliverable | Status |
|---|---|---|
| 1 | Chinese no-code strategy wizard with advanced YAML fallback | Complete |
| 2 | Strategy duplicate, version history, diff, archive, and delete flows | Next |
| 3 | Walk-forward and rolling out-of-sample backtest orchestration | Planned |
| 4 | Cost, lag, liquidity, and nearby-parameter robustness report | Planned |
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

## Delivery Order

The next implementation slice is strategy lifecycle management. Walk-forward
evaluation follows because profile recommendations should rank only candidates
that have passed credible out-of-sample and execution robustness checks.
