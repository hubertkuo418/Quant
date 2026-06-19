from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from equity_transformer.strategies.config import StrategyConfig
from equity_transformer.strategies.construction import build_target_weights
from equity_transformer.strategies.pipeline import StrategyPipeline


def make_signals(days: int = 10, tickers: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=days)
    rows = []
    for date_index, date in enumerate(dates):
        for ticker_index in range(tickers):
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{ticker_index}",
                    "score": ticker_index + date_index * 0.01,
                    "volatility": 0.1 + ticker_index * 0.1,
                    "sector": "tech" if ticker_index >= tickers // 2 else "defensive",
                }
            )
    return pd.DataFrame(rows)


def make_config(tmp_path: Path) -> StrategyConfig:
    return StrategyConfig(
        signal_path=tmp_path / "signals.parquet",
        output_path=tmp_path / "weights.parquet",
        metadata_path=tmp_path / "manifest.json",
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_only_top_k",
        top_k=2,
        long_quantile=0.25,
        short_quantile=0.25,
        weighting="equal",
        rebalance_frequency="daily",
        volatility_column="volatility",
        sector_column=None,
        max_sector_weight=None,
        max_position_weight=None,
    )


def test_long_only_top_k_weights_sum_to_one() -> None:
    weights = build_target_weights(
        make_signals(days=1),
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_only_top_k",
        top_k=2,
        rebalance_frequency="daily",
    )

    assert weights["ticker"].tolist() == ["T4", "T5"]
    assert np.isclose(weights["weight"].sum(), 1.0)
    assert set(weights["side"]) == {"long"}


def test_long_short_quantile_is_dollar_neutral() -> None:
    weights = build_target_weights(
        make_signals(days=1, tickers=8),
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_short_quantile",
        long_quantile=0.25,
        short_quantile=0.25,
        rebalance_frequency="daily",
    )

    assert np.isclose(weights["weight"].sum(), 0.0)
    assert np.isclose(weights[weights["side"] == "long"]["weight"].sum(), 0.5)
    assert np.isclose(weights[weights["side"] == "short"]["weight"].sum(), -0.5)


def test_sector_cap_limits_gross_sector_weight() -> None:
    weights = build_target_weights(
        make_signals(days=1, tickers=6),
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_only_top_k",
        top_k=3,
        rebalance_frequency="daily",
        sector_column="sector",
        max_sector_weight=0.5,
    )

    sector_gross = weights.groupby("sector")["weight"].apply(lambda x: x.abs().sum())
    assert sector_gross.max() <= 0.5
    assert "sector" in weights.columns


def test_position_cap_limits_single_name_weight() -> None:
    weights = build_target_weights(
        make_signals(days=1, tickers=6),
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_only_top_k",
        top_k=2,
        rebalance_frequency="daily",
        max_position_weight=0.4,
    )

    assert weights["weight"].abs().max() <= 0.4
    assert np.isclose(weights["weight"].sum(), 0.8)


def test_inverse_volatility_weighting_allocates_more_to_lower_risk() -> None:
    weights = build_target_weights(
        make_signals(days=1, tickers=6),
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_only_top_k",
        top_k=2,
        weighting="inverse_volatility",
        volatility_column="volatility",
        rebalance_frequency="daily",
    )

    lower_risk = weights.loc[weights["ticker"] == "T4", "weight"].iloc[0]
    higher_risk = weights.loc[weights["ticker"] == "T5", "weight"].iloc[0]
    assert lower_risk > higher_risk
    assert np.isclose(weights["weight"].sum(), 1.0)


def test_risk_parity_weighting_works_for_long_short_weights() -> None:
    weights = build_target_weights(
        make_signals(days=1, tickers=8),
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_short_quantile",
        long_quantile=0.25,
        short_quantile=0.25,
        weighting="risk_parity",
        volatility_column="volatility",
        rebalance_frequency="daily",
    )

    assert np.isclose(weights[weights["side"] == "long"]["weight"].sum(), 0.5)
    assert np.isclose(weights[weights["side"] == "short"]["weight"].sum(), -0.5)
    assert weights["weight"].abs().max() < 0.5


def test_risk_weighting_requires_volatility_column() -> None:
    try:
        build_target_weights(
            make_signals(days=1),
            date_column="date",
            ticker_column="ticker",
            score_column="score",
            strategy_type="long_only_top_k",
            top_k=2,
            weighting="risk_parity",
            rebalance_frequency="daily",
        )
    except ValueError as exc:
        assert "requires volatility_column" in str(exc)
    else:
        raise AssertionError("risk_parity should require volatility_column")


def test_sector_cap_works_for_long_short_weights() -> None:
    weights = build_target_weights(
        make_signals(days=1, tickers=8),
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_short_quantile",
        long_quantile=0.5,
        short_quantile=0.5,
        rebalance_frequency="daily",
        sector_column="sector",
        max_sector_weight=0.5,
    )

    sector_gross = weights.groupby("sector")["weight"].apply(lambda x: x.abs().sum())
    assert sector_gross.max() <= 0.5


def test_weekly_rebalance_uses_last_available_date() -> None:
    weights = build_target_weights(
        make_signals(days=10),
        date_column="date",
        ticker_column="ticker",
        score_column="score",
        strategy_type="long_only_top_k",
        top_k=1,
        rebalance_frequency="W-FRI",
    )

    assert weights["date"].nunique() == 2
    assert weights["date"].tolist() == [
        pd.Timestamp("2024-01-05"),
        pd.Timestamp("2024-01-12"),
    ]


def test_strategy_pipeline_writes_weights_and_manifest(tmp_path: Path) -> None:
    config = make_config(tmp_path)
    weights = StrategyPipeline(config).run(make_signals(days=2))

    assert len(weights) == 4
    assert config.output_path.exists()
    assert config.metadata_path.exists()

    score_config = replace(config, weighting="score", top_k=3)
    score_weights = StrategyPipeline(score_config).run(make_signals(days=1))
    assert np.isclose(score_weights["weight"].sum(), 1.0)

    sector_config = replace(config, sector_column="sector", max_sector_weight=0.5)
    sector_weights = StrategyPipeline(sector_config).run(make_signals(days=1))
    assert "sector" in sector_weights.columns

    position_config = replace(config, top_k=2, max_position_weight=0.4)
    position_weights = StrategyPipeline(position_config).run(make_signals(days=1))
    assert position_weights["weight"].abs().max() <= 0.4

    risk_config = replace(config, weighting="risk_parity", top_k=3)
    risk_weights = StrategyPipeline(risk_config).run(make_signals(days=1))
    assert np.isclose(risk_weights["weight"].sum(), 1.0)


def test_strategy_pipeline_excludes_benchmark_tickers_before_ranking(
    tmp_path: Path,
) -> None:
    config = replace(make_config(tmp_path), excluded_tickers=("T5",))

    weights = StrategyPipeline(config).run(make_signals(days=1))

    assert "T5" not in set(weights["ticker"])
    assert set(weights["ticker"]) == {"T3", "T4"}
