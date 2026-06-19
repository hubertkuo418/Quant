from __future__ import annotations

import numpy as np
import pandas as pd


def performance_metrics(
    equity_curve: pd.DataFrame,
    annualization_factor: int,
    risk_free_rate: float,
    exposure: pd.DataFrame | None = None,
    benchmark_returns: pd.Series | None = None,
) -> dict[str, float]:
    return_index = (
        pd.DatetimeIndex(pd.to_datetime(equity_curve["date"]))
        if "date" in equity_curve.columns
        else equity_curve.index
    )
    returns = pd.Series(
        equity_curve["net_return"].fillna(0.0).to_numpy(),
        index=return_index,
        name="strategy_return",
    )
    nav = equity_curve["nav"]
    periods = max(len(equity_curve) - 1, 1)
    total_return = nav.iloc[-1] / nav.iloc[0] - 1
    annual_return = (1 + total_return) ** (annualization_factor / periods) - 1
    volatility = returns.std(ddof=1) * np.sqrt(annualization_factor)
    excess_daily = returns - risk_free_rate / annualization_factor
    sharpe = (
        excess_daily.mean() / excess_daily.std(ddof=1) * np.sqrt(annualization_factor)
        if excess_daily.std(ddof=1) > 0
        else float("nan")
    )
    drawdown = nav / nav.cummax() - 1
    downside = returns[returns < 0]
    downside_volatility = downside.std(ddof=1) * np.sqrt(annualization_factor)
    sortino = (
        excess_daily.mean() * annualization_factor / downside_volatility
        if downside_volatility > 0
        else float("nan")
    )
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else float("nan")
    win_rate = float((returns > 0).mean())
    loss_tail = returns.quantile(0.05)
    tail_returns = returns[returns <= loss_tail]
    gross_profit = float(returns[returns > 0].sum())
    gross_loss = float(-returns[returns < 0].sum())
    metrics = {
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "cagr": float(annual_return),
        "annual_volatility": float(volatility),
        "sharpe_ratio": float(sharpe),
        "sortino_ratio": float(sortino),
        "max_drawdown": max_drawdown,
        "calmar_ratio": float(calmar),
        "win_rate": win_rate,
        "value_at_risk_95": float(-loss_tail),
        "conditional_value_at_risk_95": float(-tail_returns.mean()),
        "profit_factor": (
            gross_profit / gross_loss if gross_loss > 0 else float("nan")
        ),
        "average_turnover": float(equity_curve["turnover"].mean()),
        "total_cost": float(equity_curve["cost"].sum()),
    }
    if "cash_interest" in equity_curve.columns:
        metrics["total_cash_interest"] = float(equity_curve["cash_interest"].sum())
    if "borrow_cost" in equity_curve.columns:
        metrics["total_borrow_cost"] = float(equity_curve["borrow_cost"].sum())
    if exposure is not None and not exposure.empty:
        metrics.update(
            {
                "average_gross_exposure": float(exposure["gross_exposure"].mean()),
                "average_net_exposure": float(exposure["net_exposure"].mean()),
                "average_active_positions": float(exposure["active_positions"].mean()),
            }
        )
    if benchmark_returns is not None:
        metrics.update(
            relative_performance_metrics(
                returns,
                benchmark_returns,
                annualization_factor,
                risk_free_rate,
            )
        )
    return metrics


def relative_performance_metrics(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    annualization_factor: int,
    risk_free_rate: float,
) -> dict[str, float]:
    aligned = pd.concat(
        [
            strategy_returns.rename("strategy"),
            benchmark_returns.rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()
    if aligned.empty:
        return {}

    active = aligned["strategy"] - aligned["benchmark"]
    tracking_error = active.std(ddof=1) * np.sqrt(annualization_factor)
    information_ratio = (
        active.mean() / active.std(ddof=1) * np.sqrt(annualization_factor)
        if active.std(ddof=1) > 0
        else float("nan")
    )
    benchmark_variance = aligned["benchmark"].var(ddof=1)
    beta = (
        aligned["strategy"].cov(aligned["benchmark"]) / benchmark_variance
        if benchmark_variance > 0
        else float("nan")
    )
    daily_risk_free = risk_free_rate / annualization_factor
    alpha = (
        aligned["strategy"].mean()
        - daily_risk_free
        - beta * (aligned["benchmark"].mean() - daily_risk_free)
    ) * annualization_factor
    return {
        "tracking_error": float(tracking_error),
        "information_ratio": float(information_ratio),
        "beta": float(beta),
        "annual_alpha": float(alpha),
    }
