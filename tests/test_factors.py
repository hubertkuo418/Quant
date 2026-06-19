from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.factors.config import FactorValidationConfig
from equity_transformer.factors.registry import infer_factor_specs
from equity_transformer.factors.validation import FactorValidationPipeline


def make_factor_panel(days: int = 40, tickers: int = 8) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index in range(tickers):
            signal = ticker_index - (tickers - 1) / 2
            base_price = 100 + ticker_index * 10 + date_index
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{ticker_index:02d}",
                    "adj_close": base_price,
                    "return_20d": signal,
                    "volatility_20d": -signal,
                    "pe_ratio": 30 - signal,
                    "noise_factor": np.sin(date_index + ticker_index),
                    "target_5d": signal * 0.01,
                }
            )
    return pd.DataFrame(rows)


def make_config(tmp_path: Path) -> FactorValidationConfig:
    return FactorValidationConfig(
        feature_path=tmp_path / "features.parquet",
        output_dir=tmp_path / "factors",
        target_horizon=5,
        quantiles=4,
        min_cross_section=5,
        factor_columns=None,
    )


def test_factor_registry_infers_families_and_directions() -> None:
    specs = {
        spec.name: spec
        for spec in infer_factor_specs(
            ["date", "ticker", "return_20d", "pe_ratio", "volatility_20d"]
        )
    }

    assert specs["return_20d"].family == "momentum"
    assert specs["return_20d"].direction == 1
    assert specs["pe_ratio"].family == "value"
    assert specs["pe_ratio"].direction == -1
    assert specs["volatility_20d"].direction == -1
    assert "date" not in specs


def test_factor_validation_outputs_ic_and_quantile_reports(tmp_path: Path) -> None:
    outputs = FactorValidationPipeline(make_config(tmp_path)).run(make_factor_panel())

    coverage = outputs["coverage"].set_index("factor")
    ic_summary = outputs["ic_summary"].set_index("factor")
    quantiles = outputs["quantile_summary"]

    assert coverage.loc["return_20d", "coverage"] == 1.0
    assert ic_summary.loc["return_20d", "mean_rank_ic"] > 0.99
    assert ic_summary.loc["volatility_20d", "mean_rank_ic"] > 0.99

    momentum_quantiles = quantiles[quantiles["factor"] == "return_20d"]
    low = momentum_quantiles[momentum_quantiles["quantile"] == 1][
        "mean_forward_return"
    ].item()
    high = momentum_quantiles[momentum_quantiles["quantile"] == 4][
        "mean_forward_return"
    ].item()
    assert high > low
    assert (tmp_path / "factors" / "ic_summary.csv").exists()
    assert (tmp_path / "factors" / "manifest.json").exists()


def test_factor_validation_can_compute_forward_targets(tmp_path: Path) -> None:
    panel = make_factor_panel().drop(columns=["target_5d"])
    outputs = FactorValidationPipeline(make_config(tmp_path)).run(panel)

    assert not outputs["ic_summary"].empty
    assert "return_20d" in set(outputs["ic_summary"]["factor"])


def test_factor_validation_end_date_uses_target_realization_date(
    tmp_path: Path,
) -> None:
    config = make_config(tmp_path)
    config = FactorValidationConfig(
        **{**config.__dict__, "end_date": "2024-01-31"}
    )

    outputs = FactorValidationPipeline(config).run(make_factor_panel())

    assert outputs["daily_ic"]["date"].max() <= pd.Timestamp("2024-01-24")
