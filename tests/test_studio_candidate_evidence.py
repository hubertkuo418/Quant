from __future__ import annotations

from pathlib import Path

import pandas as pd

from equity_transformer.studio.candidate_evidence import (
    CandidateEvidenceConfig,
    CandidateEvidenceEvaluator,
)
from equity_transformer.studio.specs import (
    ExecutionSpec,
    PortfolioSpec,
    SignalSpec,
    StrategySpec,
    save_strategy_spec,
)


def write_candidate(tmp_path: Path, run_id: str) -> Path:
    dates = pd.bdate_range("2024-01-01", periods=55)
    market = []
    signals = []
    for ticker, drift, score in (
        ("AAA", 0.01, 1.0),
        ("BBB", -0.005, -1.0),
        ("SPY", 0.002, 0.0),
    ):
        price = 100.0
        for date in dates:
            market.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "adj_close": price,
                    "volume": 1_000_000,
                }
            )
            signals.append({"date": date, "ticker": ticker, "score": score})
            price *= 1 + drift
    market_path = tmp_path / "market.parquet"
    signal_path = tmp_path / "signals.parquet"
    pd.DataFrame(market).to_parquet(market_path, index=False)
    pd.DataFrame(signals).to_parquet(signal_path, index=False)
    run_dir = tmp_path / "runs" / run_id
    run_dir.mkdir(parents=True)
    save_strategy_spec(
        StrategySpec(
            name="candidate",
            version="1.0.0",
            description="test",
            market_path=market_path,
            signal=SignalSpec(signal_path, "score"),
            portfolio=PortfolioSpec(top_k=1, rebalance_frequency="W-FRI"),
            execution=ExecutionSpec(100_000, 1, 1, 1),
            benchmark_ticker="SPY",
        ),
        run_dir / "strategy.yaml",
    )
    return run_dir


def test_candidate_evidence_enriches_only_feasible_pareto_runs(
    tmp_path: Path,
) -> None:
    run_id = "candidate-run"
    write_candidate(tmp_path, run_id)
    optimization = tmp_path / "optimization.csv"
    pd.DataFrame(
        [
            {"run_id": run_id, "feasible": True, "pareto_efficient": True},
            {"run_id": "ignored", "feasible": True, "pareto_efficient": False},
        ]
    ).to_csv(optimization, index=False)
    config = CandidateEvidenceConfig(
        optimization_path=optimization,
        runs_root=tmp_path / "runs",
        output_dir=tmp_path / "evidence",
        scenarios=("baseline", "double_costs"),
        train_days=20,
        test_days=10,
        step_days=10,
        purge_days=2,
    )

    result = CandidateEvidenceEvaluator(config).run()

    assert result.candidates["run_id"].tolist() == [run_id]
    assert result.candidates["evidence_status"].tolist() == ["complete"]
    assert result.candidates["oos_sharpe"].notna().all()
    assert result.candidates["robustness_pass_rate"].between(0, 1).all()
    assert (result.output_dir / "evidenced_candidates.csv").exists()
