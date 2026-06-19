from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from equity_transformer.strategies.config import StrategyConfig
from equity_transformer.strategies.construction import build_target_weights


class StrategyPipeline:
    def __init__(self, config: StrategyConfig) -> None:
        self.config = config

    def run(self, signals: pd.DataFrame | None = None) -> pd.DataFrame:
        signal_frame = (
            signals.copy()
            if signals is not None
            else pd.read_parquet(self.config.signal_path)
        )
        signal_frame[self.config.date_column] = pd.to_datetime(
            signal_frame[self.config.date_column]
        )
        if self.config.start_date:
            signal_frame = signal_frame.loc[
                signal_frame[self.config.date_column]
                >= pd.Timestamp(self.config.start_date)
            ].copy()
        if self.config.end_date:
            end_date = pd.Timestamp(self.config.end_date)
            signal_frame = signal_frame.loc[
                signal_frame[self.config.date_column] <= end_date
            ].copy()
        if self.config.excluded_tickers:
            excluded = set(self.config.excluded_tickers)
            signal_frame = signal_frame.loc[
                ~signal_frame[self.config.ticker_column].str.upper().isin(excluded)
            ].copy()
        weights = build_target_weights(
            signal_frame,
            date_column=self.config.date_column,
            ticker_column=self.config.ticker_column,
            score_column=self.config.score_column,
            strategy_type=self.config.strategy_type,
            top_k=self.config.top_k,
            long_quantile=self.config.long_quantile,
            short_quantile=self.config.short_quantile,
            weighting=self.config.weighting,
            rebalance_frequency=self.config.rebalance_frequency,
            volatility_column=self.config.volatility_column,
            sector_column=self.config.sector_column,
            max_sector_weight=self.config.max_sector_weight,
            max_position_weight=self.config.max_position_weight,
        )
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        weights.to_parquet(self.config.output_path, index=False)
        self.config.metadata_path.write_text(
            json.dumps(
                {
                    "run_utc": datetime.now(UTC).isoformat(),
                    "strategy_type": self.config.strategy_type,
                    "score_column": self.config.score_column,
                    "weighting": self.config.weighting,
                    "rebalance_frequency": self.config.rebalance_frequency,
                    "volatility_column": self.config.volatility_column,
                    "sector_column": self.config.sector_column,
                    "max_sector_weight": self.config.max_sector_weight,
                    "max_position_weight": self.config.max_position_weight,
                    "excluded_tickers": list(self.config.excluded_tickers),
                    "start_date": self.config.start_date,
                    "end_date": self.config.end_date,
                    "rows": len(weights),
                    "rebalance_dates": int(weights["date"].nunique())
                    if not weights.empty
                    else 0,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return weights
