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

The Strategy tab can apply a selected factor from
`artifacts/factors/selected_factors.json` to `configs/strategy.yaml` as the
strategy `score_column`.
