from __future__ import annotations

import json

import pandas as pd

from equity_transformer.backtest.config import BacktestConfig
from equity_transformer.backtest.metrics import performance_metrics


class BacktestEngine:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.last_trade_log = pd.DataFrame()
        self.last_exposure = pd.DataFrame()
        self.last_sector_exposure = pd.DataFrame()

    def run(
        self,
        market: pd.DataFrame | None = None,
        target_weights: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
        market_frame = (
            market.copy()
            if market is not None
            else pd.read_parquet(self.config.market_path)
        )
        weights = (
            target_weights.copy()
            if target_weights is not None
            else pd.read_parquet(self.config.weights_path)
        )
        weights = self._prepare_target_weights(market_frame, weights)
        returns = self._daily_returns(market_frame)
        returns = self._trim_returns_to_strategy_start(returns, weights)
        equity_curve, holdings, trade_log, exposure, sector_exposure = self._simulate(
            returns, weights
        )
        self.last_trade_log = trade_log
        self.last_exposure = exposure
        self.last_sector_exposure = sector_exposure
        metrics = performance_metrics(
            equity_curve,
            self.config.annualization_factor,
            self.config.risk_free_rate,
            exposure,
            self._benchmark_returns(market_frame, equity_curve),
        )
        self._save(
            equity_curve,
            holdings,
            trade_log,
            exposure,
            sector_exposure,
            metrics,
        )
        return equity_curve, holdings, metrics

    @staticmethod
    def _trim_returns_to_strategy_start(
        returns: pd.DataFrame,
        target_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        if target_weights.empty:
            raise ValueError(
                "No target weights remain after execution and liquidity filters."
            )
        start_date = pd.to_datetime(target_weights["date"]).min()
        trimmed = returns.loc[returns.index >= start_date]
        if trimmed.empty:
            raise ValueError(
                "No market returns exist on or after the first target date."
            )
        return trimmed

    def _benchmark_returns(
        self,
        market: pd.DataFrame,
        equity_curve: pd.DataFrame,
    ) -> pd.Series | None:
        if self.config.benchmark_ticker is None:
            return None
        benchmark = market.loc[
            market["ticker"] == self.config.benchmark_ticker,
            ["date", "adj_close"],
        ].copy()
        if benchmark.empty:
            raise ValueError(
                f"Benchmark ticker not found: {self.config.benchmark_ticker}"
            )
        benchmark["date"] = pd.to_datetime(benchmark["date"])
        benchmark = benchmark.sort_values("date").drop_duplicates("date", keep="last")
        benchmark["benchmark_return"] = benchmark["adj_close"].pct_change()
        strategy_dates = pd.DatetimeIndex(pd.to_datetime(equity_curve["date"]))
        return benchmark.set_index("date")["benchmark_return"].reindex(strategy_dates)

    @staticmethod
    def _daily_returns(market: pd.DataFrame) -> pd.DataFrame:
        frame = market[["date", "ticker", "adj_close"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["ticker", "date"])
        frame["asset_return"] = frame.groupby("ticker")["adj_close"].pct_change()
        return (
            frame.dropna(subset=["asset_return"])
            .pivot(index="date", columns="ticker", values="asset_return")
            .sort_index()
        )

    def _prepare_target_weights(
        self,
        market: pd.DataFrame,
        target_weights: pd.DataFrame,
    ) -> pd.DataFrame:
        weights = target_weights.copy()
        weights["date"] = pd.to_datetime(weights["date"])
        if self.config.execution_lag_days:
            trading_dates = pd.Series(
                pd.to_datetime(market["date"]).drop_duplicates()
            ).sort_values(ignore_index=True)
            mapping = {
                date: trading_dates.iloc[index + self.config.execution_lag_days]
                for index, date in enumerate(trading_dates)
                if index + self.config.execution_lag_days < len(trading_dates)
            }
            weights["date"] = weights["date"].map(mapping)
            weights = weights.dropna(subset=["date"])
        if self.config.min_dollar_volume is not None:
            liquid = self._liquid_universe(market)
            weights = weights.merge(liquid, on=["date", "ticker"], how="inner")
        return weights

    def _liquid_universe(self, market: pd.DataFrame) -> pd.DataFrame:
        frame = market[["date", "ticker", "adj_close", "volume"]].copy()
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values(["ticker", "date"])
        dollar_volume = frame["adj_close"] * frame["volume"]
        frame["avg_dollar_volume"] = (
            dollar_volume.groupby(frame["ticker"])
            .rolling(
                self.config.liquidity_window,
                min_periods=self.config.liquidity_window,
            )
            .mean()
            .reset_index(level=0, drop=True)
        )
        return frame.loc[
            frame["avg_dollar_volume"] >= self.config.min_dollar_volume,
            ["date", "ticker"],
        ]

    def _simulate(
        self,
        returns: pd.DataFrame,
        target_weights: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        weights_by_date = self._weights_by_date(target_weights)
        weights_by_date = self._carry_initial_target_to_first_return(
            weights_by_date, returns.index.min()
        )
        sectors_by_ticker = self._sectors_by_ticker(target_weights)
        current_weights = pd.Series(0.0, index=returns.columns)
        nav = self.config.initial_capital
        equity_rows = []
        holding_rows = []
        trade_rows = []
        exposure_rows = []
        sector_exposure_rows = []

        first_date = returns.index.min() - pd.offsets.BDay(1)
        equity_rows.append(
            {
                "date": first_date,
                "gross_return": 0.0,
                "turnover": 0.0,
                "cost": 0.0,
                "cash_interest": 0.0,
                "borrow_cost": 0.0,
                "financing_return": 0.0,
                "net_return": 0.0,
                "nav": nav,
            }
        )

        for date, daily_returns in returns.iterrows():
            if date in weights_by_date:
                target = weights_by_date[date].reindex(returns.columns).fillna(0.0)
                delta = target - current_weights
                turnover = float(delta.abs().sum())
                cost = nav * turnover * self.config.transaction_cost_rate
                nav_after_cost = nav - cost
                for ticker, delta_weight in delta[delta != 0].items():
                    abs_delta = abs(float(delta_weight))
                    allocated_cost = cost * abs_delta / turnover if turnover else 0.0
                    trade_rows.append(
                        {
                            "date": date,
                            "ticker": ticker,
                            "from_weight": float(current_weights[ticker]),
                            "to_weight": float(target[ticker]),
                            "delta_weight": float(delta_weight),
                            "trade_value": float(nav * delta_weight),
                            "abs_trade_value": float(nav * abs_delta),
                            "cost": float(allocated_cost),
                        }
                    )
                current_weights = target
            else:
                turnover = 0.0
                cost = 0.0
                nav_after_cost = nav

            gross_return = float((current_weights * daily_returns.fillna(0.0)).sum())
            cash_weight = 1.0 - float(current_weights.sum())
            short_exposure = abs(float(current_weights[current_weights < 0].sum()))
            cash_return = (
                cash_weight
                * self.config.annual_cash_rate
                / self.config.annualization_factor
            )
            borrow_return = (
                short_exposure
                * self.config.annual_borrow_rate
                / self.config.annualization_factor
            )
            financing_return = cash_return - borrow_return
            cash_interest = nav_after_cost * cash_return
            borrow_cost = nav_after_cost * borrow_return
            nav = nav_after_cost * (1 + gross_return + financing_return)
            net_return = nav / equity_rows[-1]["nav"] - 1
            equity_rows.append(
                {
                    "date": date,
                    "gross_return": gross_return,
                    "turnover": turnover,
                    "cost": cost,
                    "cash_interest": cash_interest,
                    "borrow_cost": borrow_cost,
                    "financing_return": financing_return,
                    "net_return": net_return,
                    "nav": nav,
                }
            )
            for ticker, weight in current_weights[current_weights != 0].items():
                sector = sectors_by_ticker.get(ticker)
                holding_rows.append(
                    {
                        "date": date,
                        "ticker": ticker,
                        "sector": sector,
                        "weight": float(weight),
                        "position_value": float(nav * weight),
                    }
                )
            long_exposure = float(current_weights[current_weights > 0].sum())
            short_exposure = float(current_weights[current_weights < 0].sum())
            exposure_rows.append(
                {
                    "date": date,
                    "long_exposure": long_exposure,
                    "short_exposure": short_exposure,
                    "gross_exposure": float(current_weights.abs().sum()),
                    "net_exposure": float(current_weights.sum()),
                    "active_positions": int((current_weights != 0).sum()),
                }
            )
            sector_exposure_rows.extend(
                _sector_exposure_rows(date, current_weights, sectors_by_ticker)
            )

        equity_curve = pd.DataFrame(equity_rows).reset_index(drop=True)
        holdings = pd.DataFrame(
            holding_rows,
            columns=["date", "ticker", "sector", "weight", "position_value"],
        )
        trade_log = pd.DataFrame(
            trade_rows,
            columns=[
                "date",
                "ticker",
                "from_weight",
                "to_weight",
                "delta_weight",
                "trade_value",
                "abs_trade_value",
                "cost",
            ],
        )
        exposure = pd.DataFrame(
            exposure_rows,
            columns=[
                "date",
                "long_exposure",
                "short_exposure",
                "gross_exposure",
                "net_exposure",
                "active_positions",
            ],
        )
        sector_exposure = pd.DataFrame(
            sector_exposure_rows,
            columns=[
                "date",
                "sector",
                "long_exposure",
                "short_exposure",
                "gross_exposure",
                "net_exposure",
            ],
        )
        return equity_curve, holdings, trade_log, exposure, sector_exposure

    @staticmethod
    def _carry_initial_target_to_first_return(
        weights_by_date: dict[pd.Timestamp, pd.Series],
        first_return_date: pd.Timestamp,
    ) -> dict[pd.Timestamp, pd.Series]:
        if first_return_date in weights_by_date:
            return weights_by_date
        prior_dates = [date for date in weights_by_date if date < first_return_date]
        if not prior_dates:
            return weights_by_date
        result = dict(weights_by_date)
        result[first_return_date] = weights_by_date[max(prior_dates)]
        return result

    @staticmethod
    def _weights_by_date(target_weights: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
        weights = target_weights[["date", "ticker", "weight"]].copy()
        weights["date"] = pd.to_datetime(weights["date"])
        result = {}
        for date, daily in weights.groupby("date"):
            result[date] = daily.groupby("ticker")["weight"].sum()
        return result

    @staticmethod
    def _sectors_by_ticker(target_weights: pd.DataFrame) -> dict[str, str]:
        if "sector" not in target_weights.columns:
            return {}
        sectors = target_weights.dropna(subset=["sector"]).drop_duplicates("ticker")
        return dict(zip(sectors["ticker"], sectors["sector"], strict=False))

    def _save(
        self,
        equity_curve: pd.DataFrame,
        holdings: pd.DataFrame,
        trade_log: pd.DataFrame,
        exposure: pd.DataFrame,
        sector_exposure: pd.DataFrame,
        metrics: dict[str, float],
    ) -> None:
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        equity_curve.to_csv(self.config.output_dir / "equity_curve.csv", index=False)
        holdings.to_parquet(self.config.output_dir / "holdings.parquet", index=False)
        trade_log.to_parquet(self.config.output_dir / "trade_log.parquet", index=False)
        exposure.to_csv(self.config.output_dir / "exposure.csv", index=False)
        sector_exposure.to_csv(
            self.config.output_dir / "sector_exposure.csv", index=False
        )
        (self.config.output_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, allow_nan=True),
            encoding="utf-8",
        )


def _sector_exposure_rows(
    date: pd.Timestamp,
    weights: pd.Series,
    sectors_by_ticker: dict[str, str],
) -> list[dict[str, object]]:
    if not sectors_by_ticker:
        return []
    rows = []
    active = weights[weights != 0]
    sector_labels = pd.Series(
        [sectors_by_ticker.get(ticker, "unknown") for ticker in active.index],
        index=active.index,
        name="sector",
    )
    for sector, sector_weights in active.groupby(sector_labels, sort=True):
        rows.append(
            {
                "date": date,
                "sector": sector,
                "long_exposure": float(sector_weights[sector_weights > 0].sum()),
                "short_exposure": float(sector_weights[sector_weights < 0].sum()),
                "gross_exposure": float(sector_weights.abs().sum()),
                "net_exposure": float(sector_weights.sum()),
            }
        )
    return rows
