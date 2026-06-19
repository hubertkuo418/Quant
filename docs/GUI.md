# QuantLab Strategy Studio GUI

The initial GUI is a Streamlit app that reads existing QuantLab artifacts rather
than recalculating research logic inside the page.

## Install

```powershell
python -m pip install -e ".[dev,ml,gui]"
```

## Run

```powershell
streamlit run app/streamlit_app.py
```

## Pages

- Strategy Studio: create and validate versioned strategy specifications, run
  backtests, inspect the registry, compare runs, optimize parameters, and rank
  strategies against an investor profile.
- Dashboard: performance overview, tail-risk statistics, financing diagnostics,
  benchmark-relative metrics, equity curve, and benchmark tables.
- Factor Explorer: selected factors, IC summary, and quantile return tables.
- Strategy: target portfolio weights.
- Model Lab: model comparison, Transformer metrics, predictions, and
  strategy-ready model signals.
- Backtest: equity curve, exposure table, sector exposure, and trade log.
- Workflow: safe buttons for known QuantLab pipeline scripts and recent run logs.
- Configs: whitelist-based YAML editor for QuantLab configuration files.

## Design Rule

The GUI should remain thin. New analytics belong in tested backend modules
under `src/equity_transformer/`; Streamlit should only load artifacts, choose
filters, and display tables or charts.

Workflow buttons are backed by a whitelist in `equity_transformer.gui.workflow`.
The app cannot execute arbitrary shell commands from user input.

The config editor is also whitelist-based. It can edit known YAML files under
`configs/`, validates that the text parses to a YAML mapping, and rejects
unknown config names.

Strategy Studio specifications are stored under `strategies/`. Each execution
creates an immutable registry entry containing configuration and data hashes,
environment metadata, portfolio artifacts, and performance metrics. Comparison
and recommendation views use a common evaluation period.

## Visual Strategy Wizard

The default Strategy Studio editing mode is a Chinese no-code wizard covering:

- strategy identity and version;
- market data, signal data, and editable multi-signal components;
- universe exclusions and test dates;
- long-only or long-short construction, weighting, and rebalance frequency;
- position, sector, and liquidity limits;
- capital, costs, execution lag, financing, and benchmark assumptions.

The wizard converts every submission back into the same validated
`StrategySpec` used by CLI workflows. Changing the strategy name creates a new
slug-based YAML file; using the same name updates that strategy. Advanced YAML
mode remains available for fields that are not yet exposed in the wizard.

## Strategy Lifecycle

The Strategy Studio can duplicate strategies, snapshot overwritten versions,
compare nested StrategySpec fields, archive inactive strategies, soft-delete
with exact-name confirmation, and restore archived or deleted files. Archived
and deleted specifications are excluded from the active strategy selector.

## Walk-Forward Evaluation

The Chinese Walk-forward panel configures rolling or anchored training windows,
non-overlapping test windows, and a purge gap. It executes each OOS fold through
the normal Strategy Studio runner and displays the stitched OOS equity curve.
The current mode evaluates a frozen strategy; it does not retrain a model or
reselect factors inside every fold.

The unified robustness panel reruns the same OOS calendar under baseline,
double-cost, extra-lag, nearby Top-K, and monthly-rebalance scenarios. It shows
aggregate and worst-fold Sharpe, positive-fold rate, drawdown, and whether each
scenario passes configured constraints.

## Investor-Needs Questionnaire

The Chinese questionnaire translates risk tolerance, maximum drawdown,
turnover preference, return floor, holding period, execution conservatism, and
the desired number of recommendations into a versionable profile YAML. It also
stores desired OOS Sharpe and robustness pass-rate gates. Strict evidence mode
rejects candidates that do not provide those fields instead of silently treating
missing evidence as a pass.

The Strategy tab can apply a selected factor from
`artifacts/factors/selected_factors.json` to `configs/strategy.yaml` as the
strategy `score_column`.
