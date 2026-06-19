from __future__ import annotations

from dataclasses import dataclass

from equity_transformer.studio.recommendation import RecommendationProfile

TURNOVER_LIMITS = {
    "low": 0.15,
    "medium": 0.30,
    "flexible": 0.60,
}

EXECUTION_THRESHOLDS = {
    "strict": 0.60,
    "balanced": 0.40,
    "flexible": None,
}

OOS_THRESHOLDS = {
    "conservative": (0.50, 1.00),
    "balanced": (0.25, 0.80),
    "aggressive": (0.00, 0.60),
}


@dataclass(frozen=True)
class InvestorNeeds:
    name: str
    risk_tolerance: str
    max_drawdown: float
    turnover_preference: str
    min_annual_return: float
    holding_period: str
    execution_conservatism: str
    require_oos_evidence: bool = False
    top_n: int = 5


def profile_from_investor_needs(needs: InvestorNeeds) -> RecommendationProfile:
    if needs.risk_tolerance not in OOS_THRESHOLDS:
        raise ValueError("Unsupported risk tolerance.")
    if needs.turnover_preference not in TURNOVER_LIMITS:
        raise ValueError("Unsupported turnover preference.")
    if needs.execution_conservatism not in EXECUTION_THRESHOLDS:
        raise ValueError("Unsupported execution conservatism.")
    if needs.holding_period not in {"short", "medium", "long"}:
        raise ValueError("Unsupported holding period.")
    min_oos_sharpe, min_pass_rate = OOS_THRESHOLDS[needs.risk_tolerance]
    return RecommendationProfile(
        name=needs.name.strip(),
        risk_tolerance=needs.risk_tolerance,
        max_drawdown=needs.max_drawdown,
        max_average_turnover=TURNOVER_LIMITS[needs.turnover_preference],
        min_annual_return=needs.min_annual_return,
        min_execution_robustness=EXECUTION_THRESHOLDS[
            needs.execution_conservatism
        ],
        min_oos_sharpe=min_oos_sharpe,
        min_robustness_pass_rate=min_pass_rate,
        require_oos_evidence=needs.require_oos_evidence,
        holding_period=needs.holding_period,
        execution_conservatism=needs.execution_conservatism,
        top_n=needs.top_n,
    )
