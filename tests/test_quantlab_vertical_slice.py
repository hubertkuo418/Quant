from __future__ import annotations

from pathlib import Path

import pandas as pd

from equity_transformer.alphas.config import AlphaConfig
from equity_transformer.alphas.pipeline import AlphaPipeline
from equity_transformer.backtest.config import BacktestConfig
from equity_transformer.backtest.engine import BacktestEngine
from equity_transformer.factors.config import FactorValidationConfig
from equity_transformer.factors.panel import FactorPanelConfig, FactorPanelPipeline
from equity_transformer.factors.selection import (
    FactorSelectionConfig,
    FactorSelectionPipeline,
)
from equity_transformer.factors.signals import FactorSignalConfig, FactorSignalPipeline
from equity_transformer.factors.validation import FactorValidationPipeline
from equity_transformer.features.config import FeatureConfig
from equity_transformer.features.pipeline import FeaturePipeline
from equity_transformer.strategies.config import StrategyConfig
from equity_transformer.strategies.pipeline import StrategyPipeline


def make_market(days: int = 90, tickers: int = 6) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-02", periods=days)
    rows = []
    for ticker_index in range(tickers):
        drift = 0.05 + ticker_index * 0.02
        for day_index, date in enumerate(dates):
            close = 100 + ticker_index * 10 + drift * day_index
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{ticker_index}",
                    "open": close - 0.2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "adj_close": close,
                    "volume": 1_000 + ticker_index * 100 + day_index,
                }
            )
    return pd.DataFrame(rows)


def feature_config(tmp_path: Path) -> FeatureConfig:
    return FeatureConfig(
        market_path=tmp_path / "market.parquet",
        output_path=tmp_path / "features.parquet",
        metadata_dir=tmp_path / "metadata",
        fundamentals_path=None,
        news_path=None,
        return_windows=(1, 5, 20),
        volatility_window=20,
        rsi_window=14,
        moving_average_windows=(20, 60),
        volume_window=20,
        drop_incomplete=False,
        momentum_windows=(20,),
        volatility_windows=(20,),
        drawdown_windows=(20,),
    )


def test_quantlab_factor_to_backtest_vertical_slice(tmp_path: Path) -> None:
    market = make_market()
    features = FeaturePipeline(feature_config(tmp_path)).run(market=market)
    alpha_config = AlphaConfig(
        market_path=tmp_path / "market.parquet",
        output_path=tmp_path / "alphas.parquet",
        metadata_path=tmp_path / "alpha_manifest.json",
        alphas=("alpha_momentum_20d", "alpha_reversal_5d", "alpha101_001"),
    )
    alphas = AlphaPipeline(alpha_config).run(market)
    factor_panel = FactorPanelPipeline(
        FactorPanelConfig(
            feature_path=tmp_path / "features.parquet",
            alpha_path=tmp_path / "alphas.parquet",
            output_path=tmp_path / "factor_panel.parquet",
            metadata_path=tmp_path / "factor_panel_manifest.json",
        )
    ).run(features, alphas)
    validation = FactorValidationPipeline(
        FactorValidationConfig(
            feature_path=tmp_path / "factor_panel.parquet",
            output_dir=tmp_path / "factor_validation",
            target_horizon=5,
            quantiles=3,
            min_cross_section=5,
            factor_columns=("alpha_momentum_20d", "alpha101_001", "return_20d"),
        )
    ).run(factor_panel)
    selected = FactorSelectionPipeline(
        FactorSelectionConfig(
            ic_summary_path=tmp_path / "factor_validation" / "ic_summary.csv",
            coverage_path=tmp_path / "factor_validation" / "coverage.csv",
            output_csv_path=tmp_path / "selected_factors.csv",
            output_json_path=tmp_path / "selected_factors.json",
            min_coverage=0.5,
            min_periods=1,
            min_abs_rank_ic=0.0,
            min_positive_ic_rate=0.0,
            top_n=2,
        )
    ).run(validation["ic_summary"], validation["coverage"])
    signals = FactorSignalPipeline(
        FactorSignalConfig(
            feature_path=tmp_path / "factor_panel.parquet",
            selected_factors_path=tmp_path / "selected_factors.csv",
            output_path=tmp_path / "factor_signals.parquet",
            metadata_path=tmp_path / "factor_signals.json",
            date_column="date",
            ticker_column="ticker",
            score_column="factor_score",
            weight_column="mean_rank_ic",
            passthrough_columns=("volatility_20d",),
        )
    ).run(factor_panel, selected)
    strategy_config = StrategyConfig(
        signal_path=tmp_path / "factor_signals.parquet",
        output_path=tmp_path / "weights.parquet",
        metadata_path=tmp_path / "strategy_manifest.json",
        date_column="date",
        ticker_column="ticker",
        score_column="factor_score",
        volatility_column="volatility_20d",
        strategy_type="long_only_top_k",
        top_k=2,
        long_quantile=0.2,
        short_quantile=0.2,
        weighting="equal",
        rebalance_frequency="W-FRI",
    )
    weights = StrategyPipeline(strategy_config).run(signals)
    backtest_config = BacktestConfig(
        market_path=tmp_path / "market.parquet",
        weights_path=tmp_path / "weights.parquet",
        output_dir=tmp_path / "backtest",
        initial_capital=100_000,
        commission_bps=0,
        slippage_bps=0,
        annualization_factor=252,
        risk_free_rate=0,
    )
    equity, _, metrics = BacktestEngine(backtest_config).run(market, weights)

    assert not validation["ic_summary"].empty
    assert not selected.empty
    assert not signals.empty
    assert signals["factor_score"].notna().all()
    assert not weights.empty
    assert equity["nav"].iloc[-1] > 0
    assert "sharpe_ratio" in metrics
