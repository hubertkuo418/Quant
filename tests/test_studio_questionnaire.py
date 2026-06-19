from __future__ import annotations

import json

import pandas as pd
import pytest

from equity_transformer.studio.questionnaire import (
    InvestorNeeds,
    profile_from_investor_needs,
)
from equity_transformer.studio.recommendation import recommend_strategies


def make_candidates() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "run_id": "stable",
                "parameters": json.dumps({"portfolio.top_k": 10}),
                "annual_return": 0.15,
                "sharpe_ratio": 1.2,
                "max_drawdown": -0.08,
                "average_turnover": 0.10,
                "feasible": True,
                "oos_sharpe": 0.8,
                "robustness_pass_rate": 1.0,
            },
            {
                "run_id": "fragile",
                "parameters": json.dumps({"portfolio.top_k": 5}),
                "annual_return": 0.30,
                "sharpe_ratio": 1.8,
                "max_drawdown": -0.12,
                "average_turnover": 0.25,
                "feasible": True,
                "oos_sharpe": 0.1,
                "robustness_pass_rate": 0.5,
            },
        ]
    )


def test_questionnaire_translates_balanced_needs_to_constraints() -> None:
    profile = profile_from_investor_needs(
        InvestorNeeds(
            name="my-profile",
            risk_tolerance="balanced",
            max_drawdown=0.15,
            turnover_preference="medium",
            min_annual_return=0.05,
            holding_period="medium",
            execution_conservatism="balanced",
            require_oos_evidence=True,
        )
    )

    assert profile.max_average_turnover == 0.30
    assert profile.min_oos_sharpe == 0.25
    assert profile.min_robustness_pass_rate == 0.80
    assert profile.min_execution_robustness == 0.40


def test_recommendation_applies_oos_evidence_gates() -> None:
    profile = profile_from_investor_needs(
        InvestorNeeds(
            name="conservative",
            risk_tolerance="conservative",
            max_drawdown=0.15,
            turnover_preference="medium",
            min_annual_return=0.0,
            holding_period="long",
            execution_conservatism="flexible",
            require_oos_evidence=True,
        )
    )

    recommendations = recommend_strategies(make_candidates(), profile)

    assert recommendations["run_id"].tolist() == ["stable"]


def test_required_oos_evidence_rejects_missing_columns() -> None:
    profile = profile_from_investor_needs(
        InvestorNeeds(
            name="strict",
            risk_tolerance="balanced",
            max_drawdown=0.15,
            turnover_preference="medium",
            min_annual_return=0.0,
            holding_period="medium",
            execution_conservatism="balanced",
            require_oos_evidence=True,
        )
    )

    with pytest.raises(ValueError, match="missing oos_sharpe"):
        recommend_strategies(make_candidates().drop(columns="oos_sharpe"), profile)
