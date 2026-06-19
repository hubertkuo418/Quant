from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from equity_transformer.reporting.config import ReportConfig

MODEL_COMPARISON_COLUMNS = [
    "model",
    "horizon",
    "mae",
    "rmse",
    "pearson_ic",
    "rank_ic",
    "directional_accuracy",
    "source",
]


class ReportPipeline:
    def __init__(self, config: ReportConfig) -> None:
        self.config = config

    def run(self) -> dict[str, pd.DataFrame]:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        model_metrics = build_model_comparison(
            self.config.baselines_metrics_path,
            self.config.recurrent_metrics_path,
            self.config.transformer_metrics_path,
        )
        portfolio_metrics = build_portfolio_comparison(
            self.config.backtest_metrics_path,
            self.config.benchmark_comparison_path,
        )
        trading_diagnostics = build_trading_diagnostics(
            self.config.trade_log_path,
            self.config.exposure_path,
            self.config.sector_exposure_path,
        )
        model_metrics.to_csv(
            self.config.output_dir / "model_comparison.csv", index=False
        )
        portfolio_metrics.to_csv(
            self.config.output_dir / "portfolio_comparison.csv", index=False
        )
        trading_diagnostics.to_csv(
            self.config.output_dir / "trading_diagnostics.csv", index=False
        )
        return {
            "model_comparison": model_metrics,
            "portfolio_comparison": portfolio_metrics,
            "trading_diagnostics": trading_diagnostics,
        }


def build_model_comparison(
    baselines_metrics_path: str | Path,
    recurrent_metrics_path: str | Path,
    transformer_metrics_path: str | Path,
) -> pd.DataFrame:
    frames = []
    baseline_path = Path(baselines_metrics_path)
    if baseline_path.exists():
        frames.append(pd.read_csv(baseline_path).assign(source="baseline"))

    recurrent_path = Path(recurrent_metrics_path)
    if recurrent_path.exists():
        recurrent = pd.read_csv(recurrent_path)
        if "model" not in recurrent.columns:
            recurrent = recurrent.assign(model="recurrent")
        frames.append(recurrent.assign(source="recurrent"))

    transformer_path = Path(transformer_metrics_path)
    if transformer_path.exists():
        payload = _read_json(transformer_path)
        metrics_by_horizon = payload.get("metrics_by_horizon", {})
        for horizon, metrics in metrics_by_horizon.items():
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "model": "transformer_v1",
                            "horizon": int(horizon),
                            **metrics,
                            "source": "transformer",
                        }
                    ]
                )
            )
        metrics = payload.get("metrics", {})
        if metrics and not metrics_by_horizon:
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "model": "transformer_v1",
                            "horizon": payload.get("target_horizon"),
                            **metrics,
                            "source": "transformer",
                        }
                    ]
                )
            )
    if not frames:
        return pd.DataFrame(columns=MODEL_COMPARISON_COLUMNS)
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(
        ["horizon", "rmse", "model"],
        na_position="last",
    )


def build_portfolio_comparison(
    backtest_metrics_path: str | Path,
    benchmark_comparison_path: str | Path,
) -> pd.DataFrame:
    frames = []
    backtest_path = Path(backtest_metrics_path)
    if backtest_path.exists():
        frames.append(
            pd.DataFrame([{**_read_json(backtest_path), "portfolio": "strategy"}])
        )
    benchmark_path = Path(benchmark_comparison_path)
    if benchmark_path.exists():
        benchmarks = pd.read_csv(benchmark_path).rename(
            columns={"benchmark": "portfolio"}
        )
        frames.append(benchmarks)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False).sort_values(
        "sharpe_ratio",
        ascending=False,
        na_position="last",
    )


def build_trading_diagnostics(
    trade_log_path: str | Path,
    exposure_path: str | Path,
    sector_exposure_path: str | Path | None = None,
) -> pd.DataFrame:
    trade_path = Path(trade_log_path)
    exposure_file = Path(exposure_path)
    sector_exposure_file = (
        Path(sector_exposure_path) if sector_exposure_path is not None else None
    )
    rows = {}
    if trade_path.exists():
        trades = pd.read_parquet(trade_path)
        rows.update(
            {
                "trade_count": float(len(trades)),
                "total_abs_trade_value": float(trades["abs_trade_value"].sum()),
                "total_trade_cost": float(trades["cost"].sum()),
                "average_abs_trade_value": float(trades["abs_trade_value"].mean())
                if not trades.empty
                else 0.0,
            }
        )
    if exposure_file.exists():
        exposure = pd.read_csv(exposure_file)
        rows.update(
            {
                "average_gross_exposure": float(exposure["gross_exposure"].mean()),
                "average_net_exposure": float(exposure["net_exposure"].mean()),
                "max_gross_exposure": float(exposure["gross_exposure"].max()),
                "average_active_positions": float(exposure["active_positions"].mean()),
            }
        )
    if sector_exposure_file is not None and sector_exposure_file.exists():
        sector_exposure = pd.read_csv(sector_exposure_file)
        rows.update(_sector_exposure_diagnostics(sector_exposure))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([rows])


def _sector_exposure_diagnostics(sector_exposure: pd.DataFrame) -> dict[str, float]:
    if sector_exposure.empty:
        return {}
    by_sector = sector_exposure.groupby("sector")["gross_exposure"].mean()
    return {
        "max_sector_gross_exposure": float(sector_exposure["gross_exposure"].max()),
        "average_largest_sector_gross_exposure": float(
            sector_exposure.groupby("date")["gross_exposure"].max().mean()
        ),
        "sector_count": float(sector_exposure["sector"].nunique()),
        "largest_average_sector_gross_exposure": float(by_sector.max()),
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
