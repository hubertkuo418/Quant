# Formulaic Alphas

QuantLab now has a formulaic alpha framework:

- Alpha definitions live in a registry.
- Each alpha has a name, family, lookback, description, and function.
- Batch calculation writes a dated `(date, ticker)` alpha panel.
- Metadata records coverage for every alpha.

## Current Starter Alphas

```text
alpha_momentum_20d
alpha_reversal_5d
alpha_volume_price_corr_20d
alpha_volatility_reversal_20d
alpha_price_position_20d
```

## Alpha101-Compatible Batch

The registry includes the first 20 named Alpha101-compatible formulaic
signals. These are transparent research proxies inspired by common Alpha101
operators and economic intuitions; they are not claimed to be verbatim
reproductions of proprietary or publication formulas.

```text
alpha101_001
alpha101_002
alpha101_003
alpha101_004
alpha101_005
alpha101_006
alpha101_007
alpha101_008
alpha101_009
alpha101_010
alpha101_011
alpha101_012
alpha101_013
alpha101_014
alpha101_015
alpha101_016
alpha101_017
alpha101_018
alpha101_019
alpha101_020
```

The 011-020 batch adds:

| Alpha | Main signal | Lookback |
| --- | --- | ---: |
| 011 | Close deviation from trailing VWAP proxy | 10 |
| 012 | Price reversal signed by volume direction | 1 |
| 013 | Ranked close-volume covariance | 5 |
| 014 | Return rank scaled by open-volume correlation | 10 |
| 015 | Summed ranked high-volume correlation | 5 |
| 016 | Ranked high-volume covariance | 5 |
| 017 | Price position, acceleration, and volume interaction | 20 |
| 018 | Intraday spread and open-close correlation | 10 |
| 019 | Direction reversal scaled by long momentum | 60 |
| 020 | Ranked overnight OHLC gap product | 1 |

This is not yet the full 101-alpha library. The important design choice is
that future formulas can be added by registering new `AlphaDefinition` entries
without changing the downstream factor validation, strategy, backtest, or GUI
contracts.

## Workflow

```powershell
python scripts/build_alphas.py --config configs/alphas.yaml
```

Outputs:

```text
data/processed/alpha_panel.parquet
data/metadata/alpha_manifest.json
```

Merge alphas into the unified factor panel:

```powershell
python scripts/build_factor_panel.py --config configs/factor_panel.yaml
```

Output:

```text
data/processed/factor_panel.parquet
```
