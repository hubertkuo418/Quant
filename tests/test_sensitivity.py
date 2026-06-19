from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from equity_transformer.backtest.config import BacktestConfig
from equity_transformer.backtest.sensitivity import (
    SensitivityConfig,
    SensitivityPipeline,
    SensitivityScenario,
)


def make_market() -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=8)
    rows = []
    for ticker, drift in (("AAA", 0.02), ("BBB", 0.0)):
        for index, date in enumerate(dates):
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": 100 * (1 + drift) ** index,
                }
            )
    return pd.DataFrame(rows)


def make_weights() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-05")],
            "ticker": ["AAA", "AAA"],
            "weight": [1.0, 0.5],
        }
    )


def make_base_config(tmp_path: Path) -> BacktestConfig:
    return BacktestConfig(
        market_path=tmp_path / "market.parquet",
        weights_path=tmp_path / "weights.parquet",
        output_dir=tmp_path / "base",
        initial_capital=100.0,
        commission_bps=5.0,
        slippage_bps=5.0,
        annualization_factor=252,
        risk_free_rate=0.0,
    )


def test_sensitivity_pipeline_compares_scenarios(tmp_path: Path) -> None:
    config = SensitivityConfig(
        base_backtest_config=tmp_path / "backtest.yaml",
        output_dir=tmp_path / "sensitivity",
        scenarios=(
            SensitivityScenario("baseline", {}),
            SensitivityScenario(
                "double_costs", {"commission_bps": 10, "slippage_bps": 10}
            ),
            SensitivityScenario("lag_1d", {"execution_lag_days": 1}),
        ),
    )

    comparison = SensitivityPipeline(config).run(
        make_market(),
        make_weights(),
        make_base_config(tmp_path),
    )

    indexed = comparison.set_index("scenario")
    assert set(indexed.index) == {"baseline", "double_costs", "lag_1d"}
    assert indexed.loc["double_costs", "ending_nav"] < indexed.loc[
        "baseline", "ending_nav"
    ]
    assert (config.output_dir / "comparison.csv").exists()
    assert (config.output_dir / "manifest.json").exists()
    assert (config.output_dir / "baseline" / "metrics.json").exists()


@pytest.mark.parametrize(
    ("scenario", "message"),
    [
        (SensitivityScenario("../escape", {}), "Invalid sensitivity"),
        (SensitivityScenario("bad", {"initial_capital": 1}), "unsupported"),
        (SensitivityScenario("bad", {"execution_lag_days": -1}), "negative"),
    ],
)
def test_sensitivity_pipeline_rejects_invalid_scenarios(
    tmp_path: Path,
    scenario: SensitivityScenario,
    message: str,
) -> None:
    config = SensitivityConfig(
        base_backtest_config=tmp_path / "backtest.yaml",
        output_dir=tmp_path / "sensitivity",
        scenarios=(scenario,),
    )

    with pytest.raises(ValueError, match=message):
        SensitivityPipeline(config).run(
            make_market(), make_weights(), make_base_config(tmp_path)
        )
