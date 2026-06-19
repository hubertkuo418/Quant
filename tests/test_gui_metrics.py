from __future__ import annotations

from equity_transformer.gui.metrics import (
    RELATIVE_METRICS,
    TAIL_RISK_METRICS,
    available_metrics,
    metric_table,
)


def test_available_metrics_skips_missing_and_non_numeric_values() -> None:
    metrics = {"sharpe_ratio": 1.25, "annual_return": "missing"}

    available = available_metrics(
        metrics, ("annual_return", "sharpe_ratio", "max_drawdown")
    )

    assert available == [("sharpe_ratio", 1.25)]


def test_metric_table_labels_available_tail_risk_metrics() -> None:
    table = metric_table(
        {
            "value_at_risk_95": 0.03,
            "conditional_value_at_risk_95": 0.05,
            "profit_factor": 1.4,
        },
        TAIL_RISK_METRICS,
    )

    assert table["metric"].tolist() == [
        "Value At Risk 95",
        "Conditional Value At Risk 95",
        "Profit Factor",
    ]


def test_metric_table_is_empty_without_relative_benchmark_metrics() -> None:
    table = metric_table({"sharpe_ratio": 1.0}, RELATIVE_METRICS)

    assert table.empty
    assert table.columns.tolist() == ["metric", "value"]
