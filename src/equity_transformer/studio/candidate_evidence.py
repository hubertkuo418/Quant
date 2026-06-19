from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import yaml

from equity_transformer.studio.robustness import (
    KNOWN_SCENARIOS,
    RobustnessConfig,
    RobustnessEvaluator,
)


@dataclass(frozen=True)
class CandidateEvidenceConfig:
    optimization_path: Path
    runs_root: Path
    output_dir: Path
    scenarios: tuple[str, ...] = KNOWN_SCENARIOS
    train_days: int = 120
    test_days: int = 20
    step_days: int = 20
    purge_days: int = 5
    max_candidates: int = 10


@dataclass(frozen=True)
class CandidateEvidenceResult:
    output_dir: Path
    candidates: pd.DataFrame


def load_candidate_evidence_config(path: str | Path) -> CandidateEvidenceConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    return CandidateEvidenceConfig(
        optimization_path=Path(payload["optimization_path"]),
        runs_root=Path(payload.get("runs_root", "artifacts/studio/runs")),
        output_dir=Path(payload["output_dir"]),
        scenarios=tuple(payload.get("scenarios", KNOWN_SCENARIOS)),
        train_days=int(payload.get("train_days", 120)),
        test_days=int(payload.get("test_days", 20)),
        step_days=int(payload.get("step_days", payload.get("test_days", 20))),
        purge_days=int(payload.get("purge_days", 5)),
        max_candidates=int(payload.get("max_candidates", 10)),
    )


class CandidateEvidenceEvaluator:
    def __init__(self, config: CandidateEvidenceConfig) -> None:
        self.config = config
        if config.max_candidates <= 0:
            raise ValueError("max_candidates must be positive.")
        if "baseline" not in config.scenarios:
            raise ValueError("Candidate evidence scenarios must include baseline.")

    def run(self) -> CandidateEvidenceResult:
        candidates = pd.read_csv(self.config.optimization_path)
        required = {"run_id", "feasible", "pareto_efficient"}
        missing = required.difference(candidates.columns)
        if missing:
            raise ValueError(f"Optimization results missing: {sorted(missing)}")
        selected = candidates.loc[
            candidates["feasible"].map(_as_bool)
            & candidates["pareto_efficient"].map(_as_bool)
        ].head(self.config.max_candidates)
        if selected.empty:
            raise ValueError("No feasible Pareto candidates require evidence.")

        output = self.config.output_dir
        output.mkdir(parents=True, exist_ok=True)
        evidence_rows = []
        for row in selected.itertuples(index=False):
            run_id = str(row.run_id)
            spec_path = self.config.runs_root / run_id / "strategy.yaml"
            if not spec_path.exists():
                raise FileNotFoundError(f"Candidate StrategySpec missing: {spec_path}")
            result = RobustnessEvaluator(
                RobustnessConfig(
                    strategy_spec_path=spec_path,
                    output_dir=output / "candidates" / run_id,
                    scenarios=self.config.scenarios,
                    train_days=self.config.train_days,
                    test_days=self.config.test_days,
                    step_days=self.config.step_days,
                    purge_days=self.config.purge_days,
                )
            ).run()
            baseline = result.scenarios.loc[
                result.scenarios["scenario"] == "baseline"
            ]
            if baseline.empty:
                raise ValueError("Candidate evidence scenarios must include baseline.")
            baseline_row = baseline.iloc[0]
            evidence_rows.append(
                {
                    "run_id": run_id,
                    "oos_observations": int(baseline_row["oos_observations"]),
                    "oos_total_return": float(baseline_row["total_return"]),
                    "oos_sharpe": float(baseline_row["sharpe_ratio"]),
                    "oos_max_drawdown": float(baseline_row["max_drawdown"]),
                    "robustness_pass_rate": float(result.aggregate["pass_rate"]),
                    "robustness_worst_sharpe": float(
                        result.aggregate["worst_sharpe"]
                    ),
                    "robustness_worst_drawdown": float(
                        result.aggregate["worst_max_drawdown"]
                    ),
                    "evidence_status": "complete",
                }
            )
        evidence = pd.DataFrame(evidence_rows)
        enriched = selected.merge(evidence, on="run_id", how="left", validate="1:1")
        enriched.to_csv(output / "evidenced_candidates.csv", index=False)
        manifest = {
            "created_utc": datetime.now(UTC).isoformat(),
            "optimization_path": str(self.config.optimization_path),
            "candidate_count": len(enriched),
            "scenarios": list(self.config.scenarios),
            "evaluation": "per_candidate_frozen_strategy_oos",
        }
        (output / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return CandidateEvidenceResult(output, enriched)


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes"}
