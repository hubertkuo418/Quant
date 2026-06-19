from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from equity_transformer.data.validation import MARKET_COLUMNS, validate_market_frame
from equity_transformer.features.config import FeatureConfig
from equity_transformer.features.fundamentals import (
    FUNDAMENTAL_COLUMNS,
    merge_point_in_time_fundamentals,
)
from equity_transformer.features.sentiment import merge_news_features
from equity_transformer.features.technical import build_technical_features


class FeaturePipeline:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config

    def run(
        self,
        market: pd.DataFrame | None = None,
        fundamentals: pd.DataFrame | None = None,
        news: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        market_frame = (
            market.copy()
            if market is not None
            else pd.read_parquet(self.config.market_path)
        )
        validate_market_frame(market_frame)
        features = build_technical_features(market_frame, self.config)

        fundamental_frame = (
            fundamentals
            if fundamentals is not None
            else self._read_optional(self.config.fundamentals_path)
        )
        if fundamental_frame is not None:
            features = merge_point_in_time_fundamentals(features, fundamental_frame)
        else:
            for column in FUNDAMENTAL_COLUMNS:
                features[column] = float("nan")
            features["fundamental_available_at"] = pd.NaT

        news_frame = (
            news
            if news is not None
            else self._read_optional(self.config.news_path)
        )
        if news_frame is not None:
            features = merge_news_features(
                features,
                news_frame,
                market_timezone=self.config.news_market_timezone,
                cutoff_time=self.config.news_cutoff_time,
            )
        else:
            features["news_sentiment"] = 0.0
            features["news_article_count"] = 0

        feature_columns = [
            column
            for column in features.columns
            if column not in MARKET_COLUMNS and column != "fundamental_available_at"
        ]
        if self.config.drop_incomplete:
            features = features.dropna(subset=feature_columns)

        features = features.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)
        features.to_parquet(self.config.output_path, index=False)
        self._write_coverage_report(features, feature_columns)
        return features

    @staticmethod
    def _read_optional(path: Path | None) -> pd.DataFrame | None:
        if path is None:
            return None
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)

    def _write_coverage_report(
        self, features: pd.DataFrame, feature_columns: list[str]
    ) -> None:
        run_time = datetime.now(UTC)
        report = {
            "run_utc": run_time.isoformat(),
            "rows": len(features),
            "tickers": features["ticker"].nunique(),
            "min_date": features["date"].min().isoformat(),
            "max_date": features["date"].max().isoformat(),
            "feature_columns": feature_columns,
            "coverage": {
                column: float(features[column].notna().mean())
                for column in feature_columns
            },
        }
        path = self.config.metadata_dir / f"features_{run_time:%Y%m%dT%H%M%SZ}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
