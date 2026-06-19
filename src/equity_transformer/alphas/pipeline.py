from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd

from equity_transformer.alphas.config import AlphaConfig
from equity_transformer.alphas.registry import alpha_registry
from equity_transformer.data.validation import validate_market_frame


class AlphaPipeline:
    def __init__(self, config: AlphaConfig) -> None:
        self.config = config

    def run(self, market: pd.DataFrame | None = None) -> pd.DataFrame:
        frame = (
            market.copy()
            if market is not None
            else pd.read_parquet(self.config.market_path)
        )
        validate_market_frame(frame)
        frame = frame.sort_values(["ticker", "date"]).reset_index(drop=True)
        registry = alpha_registry()
        missing = [name for name in self.config.alphas if name not in registry]
        if missing:
            raise ValueError(f"Unknown alphas: {missing}")

        output = frame[["date", "ticker"]].copy()
        manifest_alphas = []
        for name in self.config.alphas:
            definition = registry[name]
            output[name] = definition.function(frame)
            manifest_alphas.append(
                {
                    "name": definition.name,
                    "family": definition.family,
                    "lookback": definition.lookback,
                    "description": definition.description,
                    "coverage": float(output[name].notna().mean()),
                }
            )

        output = output.sort_values(["date", "ticker"]).reset_index(drop=True)
        self.config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(self.config.output_path, index=False)
        self.config.metadata_path.write_text(
            json.dumps(
                {
                    "run_utc": datetime.now(UTC).isoformat(),
                    "rows": len(output),
                    "alphas": manifest_alphas,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return output
