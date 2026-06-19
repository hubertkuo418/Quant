from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

PREFERENCE_WEIGHTS = {
    "conservative": {
        "annual_return": 0.10,
        "sharpe_ratio": 0.20,
        "max_drawdown": 0.35,
        "average_turnover": 0.15,
        "execution_robustness": 0.20,
    },
    "balanced": {
        "annual_return": 0.20,
        "sharpe_ratio": 0.30,
        "max_drawdown": 0.20,
        "average_turnover": 0.10,
        "execution_robustness": 0.20,
    },
    "aggressive": {
        "annual_return": 0.35,
        "sharpe_ratio": 0.30,
        "max_drawdown": 0.10,
        "average_turnover": 0.05,
        "execution_robustness": 0.20,
    },
}


@dataclass(frozen=True)
class RecommendationProfile:
    name: str
    risk_tolerance: str = "balanced"
    max_drawdown: float | None = None
    max_average_turnover: float | None = None
    min_annual_return: float | None = None
    min_execution_robustness: float | None = None
    min_oos_sharpe: float | None = None
    min_robustness_pass_rate: float | None = None
    require_oos_evidence: bool = False
    holding_period: str = "medium"
    execution_conservatism: str = "balanced"
    top_n: int = 5


def load_recommendation_profile(path: str | Path) -> RecommendationProfile:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream) or {}
    profile = RecommendationProfile(
        name=str(payload.get("name", "default")),
        risk_tolerance=str(payload.get("risk_tolerance", "balanced")),
        max_drawdown=_optional_float(payload.get("max_drawdown")),
        max_average_turnover=_optional_float(payload.get("max_average_turnover")),
        min_annual_return=_optional_float(payload.get("min_annual_return")),
        min_execution_robustness=_optional_float(
            payload.get("min_execution_robustness")
        ),
        min_oos_sharpe=_optional_float(payload.get("min_oos_sharpe")),
        min_robustness_pass_rate=_optional_float(
            payload.get("min_robustness_pass_rate")
        ),
        require_oos_evidence=bool(payload.get("require_oos_evidence", False)),
        holding_period=str(payload.get("holding_period", "medium")),
        execution_conservatism=str(
            payload.get("execution_conservatism", "balanced")
        ),
        top_n=int(payload.get("top_n", 5)),
    )
    _validate_profile(profile)
    return profile


def recommend_strategies(
    candidates: pd.DataFrame,
    profile: RecommendationProfile,
) -> pd.DataFrame:
    _validate_profile(profile)
    required = {
        "run_id",
        "parameters",
        "annual_return",
        "sharpe_ratio",
        "max_drawdown",
        "average_turnover",
    }
    missing = required.difference(candidates.columns)
    if missing:
        raise ValueError(f"Recommendation candidates missing: {sorted(missing)}")
    frame = candidates.copy()
    if "feasible" in frame.columns:
        frame = frame[frame["feasible"].map(_as_bool)]
    if profile.max_drawdown is not None:
        frame = frame[frame["max_drawdown"] >= -profile.max_drawdown]
    if profile.max_average_turnover is not None:
        frame = frame[frame["average_turnover"] <= profile.max_average_turnover]
    if profile.min_annual_return is not None:
        frame = frame[frame["annual_return"] >= profile.min_annual_return]
    frame = _apply_oos_evidence_gates(frame, profile)
    if frame.empty:
        raise ValueError("No strategy candidates satisfy the user profile constraints.")

    frame["strategy_family"] = frame["parameters"].map(_strategy_family)
    family_stats = frame.groupby("strategy_family")["sharpe_ratio"].agg(
        worst_case_sharpe="min",
        lag_sharpe_std="std",
    )
    family_stats["lag_sharpe_std"] = family_stats["lag_sharpe_std"].fillna(0.0)
    frame = frame.join(family_stats, on="strategy_family")
    frame["execution_robustness"] = (
        frame["worst_case_sharpe"].rank(pct=True)
        + (1 / (1 + frame["lag_sharpe_std"])).rank(pct=True)
    ) / 2
    if profile.min_execution_robustness is not None:
        frame = frame[
            frame["execution_robustness"] >= profile.min_execution_robustness
        ]
    if frame.empty:
        raise ValueError("No strategy candidates satisfy robustness constraints.")

    weights = PREFERENCE_WEIGHTS[profile.risk_tolerance]
    frame["recommendation_score"] = (
        weights["annual_return"] * frame["annual_return"].rank(pct=True)
        + weights["sharpe_ratio"] * frame["sharpe_ratio"].rank(pct=True)
        + weights["max_drawdown"] * frame["max_drawdown"].rank(pct=True)
        + weights["average_turnover"] * (-frame["average_turnover"]).rank(pct=True)
        + weights["execution_robustness"] * frame["execution_robustness"]
    )
    frame = frame.sort_values(
        ["recommendation_score", "sharpe_ratio"], ascending=False
    ).reset_index(drop=True)
    frame["recommendation_rank"] = frame.index + 1
    frame["profile"] = profile.name
    frame["rationale"] = frame.apply(
        lambda row: (
            f"{profile.risk_tolerance} 需求；Sharpe {row['sharpe_ratio']:.2f}，"
            f"回撤 {row['max_drawdown']:.1%}，換手率 "
            f"{row['average_turnover']:.1%}，最差延遲 Sharpe "
            f"{row['worst_case_sharpe']:.2f}"
        ),
        axis=1,
    )
    return frame.head(profile.top_n)


def save_recommendations(
    recommendations: pd.DataFrame,
    profile: RecommendationProfile,
    output_dir: str | Path,
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output / "recommendations.csv", index=False)
    (output / "profile.json").write_text(
        json.dumps(profile.__dict__, indent=2), encoding="utf-8"
    )


def save_recommendation_profile(
    profile: RecommendationProfile,
    path: str | Path,
) -> None:
    _validate_profile(profile)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(profile.__dict__, sort_keys=False),
        encoding="utf-8",
    )


def _strategy_family(parameters: str) -> str:
    payload = json.loads(parameters)
    payload.pop("execution.execution_lag_days", None)
    return json.dumps(payload, sort_keys=True)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}


def _optional_float(value: object) -> float | None:
    return float(value) if value is not None else None


def _validate_profile(profile: RecommendationProfile) -> None:
    if profile.risk_tolerance not in PREFERENCE_WEIGHTS:
        raise ValueError(
            "risk_tolerance must be conservative, balanced, or aggressive."
        )
    if profile.max_drawdown is not None and not 0 < profile.max_drawdown <= 1:
        raise ValueError("max_drawdown must be in (0, 1].")
    if (
        profile.max_average_turnover is not None
        and not 0 <= profile.max_average_turnover <= 1
    ):
        raise ValueError("max_average_turnover must be in [0, 1].")
    if profile.top_n <= 0:
        raise ValueError("top_n must be positive.")
    for name in (
        "min_execution_robustness",
        "min_robustness_pass_rate",
    ):
        value = getattr(profile, name)
        if value is not None and not 0 <= value <= 1:
            raise ValueError(f"{name} must be in [0, 1].")


def _apply_oos_evidence_gates(
    frame: pd.DataFrame,
    profile: RecommendationProfile,
) -> pd.DataFrame:
    requirements = {
        "oos_sharpe": profile.min_oos_sharpe,
        "robustness_pass_rate": profile.min_robustness_pass_rate,
    }
    result = frame
    for column, minimum in requirements.items():
        if minimum is None:
            continue
        if column not in result.columns:
            if profile.require_oos_evidence:
                raise ValueError(f"Recommendation candidates missing {column}.")
            continue
        result = result[result[column] >= minimum]
    return result
