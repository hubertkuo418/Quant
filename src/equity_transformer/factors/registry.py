from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorSpec:
    name: str
    family: str
    direction: int
    description: str


EXCLUDED_COLUMNS = {
    "date",
    "ticker",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "fundamental_available_at",
    "news_article_count",
    "split",
}


def infer_factor_specs(columns: list[str] | tuple[str, ...]) -> list[FactorSpec]:
    specs = []
    for column in columns:
        if _is_factor_column(column):
            specs.append(
                FactorSpec(
                    name=column,
                    family=_infer_family(column),
                    direction=_infer_direction(column),
                    description=_infer_description(column),
                )
            )
    return specs


def _is_factor_column(column: str) -> bool:
    if column in EXCLUDED_COLUMNS:
        return False
    if column.startswith(("target_", "target_date_")):
        return False
    return True


def _infer_family(column: str) -> str:
    lower = column.lower()
    if lower.startswith("alpha"):
        return "alpha101"
    if "return" in lower or "momentum" in lower:
        return "momentum"
    if lower in {"pe_ratio", "pb_ratio"} or any(
        token in lower for token in ["earnings_yield", "ev_ebitda", "ps_ratio"]
    ):
        return "value"
    if any(token in lower for token in ["roe", "roa", "margin", "debt"]):
        return "quality"
    if any(token in lower for token in ["volatility", "drawdown", "beta"]):
        return "volatility"
    if any(
        token in lower
        for token in ["rsi", "ma_", "macd", "sma", "ema", "atr", "bollinger"]
    ):
        return "technical"
    if "sentiment" in lower or "news" in lower:
        return "news"
    if "volume" in lower or "liquidity" in lower:
        return "liquidity"
    return "other"


def _infer_direction(column: str) -> int:
    lower = column.lower()
    if lower in {"pe_ratio", "pb_ratio", "ps_ratio", "ev_ebitda"}:
        return -1
    if "volatility" in lower or "drawdown" in lower or "debt" in lower:
        return -1
    return 1


def _infer_description(column: str) -> str:
    family = _infer_family(column)
    direction = "higher is expected to be better" if _infer_direction(column) > 0 else (
        "lower is expected to be better"
    )
    return f"{family} factor inferred from column name; {direction}."
