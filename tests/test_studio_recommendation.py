from __future__ import annotations

import json

import pandas as pd
import pytest

from equity_transformer.studio.recommendation import (
    RecommendationProfile,
    recommend_strategies,
)


def make_candidates() -> pd.DataFrame:
    rows = []
    for top_k, lag, sharpe, drawdown, turnover, annual_return in (
        (5, 1, 2.0, -0.20, 0.30, 0.40),
        (5, 2, 0.2, -0.18, 0.28, 0.10),
        (10, 1, 1.2, -0.08, 0.12, 0.20),
        (10, 2, 1.0, -0.09, 0.13, 0.18),
    ):
        rows.append(
            {
                "run_id": f"top-{top_k}-lag-{lag}",
                "parameters": json.dumps(
                    {
                        "portfolio.top_k": top_k,
                        "execution.execution_lag_days": lag,
                    }
                ),
                "annual_return": annual_return,
                "sharpe_ratio": sharpe,
                "max_drawdown": drawdown,
                "average_turnover": turnover,
                "feasible": True,
            }
        )
    return pd.DataFrame(rows)


def test_balanced_profile_prefers_stable_candidate_family() -> None:
    profile = RecommendationProfile(
        name="balanced",
        risk_tolerance="balanced",
        max_drawdown=0.15,
        max_average_turnover=0.2,
        top_n=2,
    )

    recommendations = recommend_strategies(make_candidates(), profile)

    assert recommendations["run_id"].str.startswith("top-10").all()
    assert recommendations["recommendation_rank"].tolist() == [1, 2]
    assert recommendations["rationale"].str.contains("最差延遲 Sharpe").all()


def test_profile_reports_when_constraints_remove_every_candidate() -> None:
    profile = RecommendationProfile(
        name="impossible",
        max_drawdown=0.01,
    )

    with pytest.raises(ValueError, match="No strategy candidates"):
        recommend_strategies(make_candidates(), profile)
