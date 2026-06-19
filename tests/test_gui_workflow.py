from __future__ import annotations

from pathlib import Path

import pytest

from equity_transformer.gui import workflow
from equity_transformer.gui.workflow import (
    WorkflowStep,
    latest_workflow_runs,
    list_workflow_steps,
    run_workflow_step,
)


def test_list_workflow_steps_exposes_known_steps() -> None:
    names = {step.name for step in list_workflow_steps()}

    assert {
        "csv_import",
        "studio_strategy",
        "studio_optimizer",
        "studio_recommend",
        "studio_report",
        "studio_walk_forward",
        "data_quality",
        "features",
        "alphas",
        "catalog",
        "backtest",
        "regime_analysis",
        "sensitivity",
        "attribution",
        "equity_comparison",
        "factor_signals",
        "transformer",
        "prediction_signals",
        "model_strategy",
        "model_backtest",
    }.issubset(names)


def test_run_workflow_step_records_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    script = tmp_path / "ok.py"
    script.write_text("print('hello workflow')\n", encoding="utf-8")
    monkeypatch.setitem(
        workflow.WORKFLOW_STEPS,
        "ok",
        WorkflowStep(
            name="ok",
            label="OK",
            command=(script.name,),
            description="temporary successful step",
        ),
    )

    result = run_workflow_step("ok", root=tmp_path)
    runs = latest_workflow_runs(tmp_path)

    assert result["success"] is True
    assert "hello workflow" in result["stdout"]
    assert len(runs) == 1
    assert runs[0]["step"] == "ok"


def test_run_workflow_step_rejects_unknown_step(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown workflow step"):
        run_workflow_step("not_registered", root=tmp_path)
