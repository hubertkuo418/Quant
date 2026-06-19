from __future__ import annotations

import argparse

from equity_transformer.studio.candidate_evidence import (
    CandidateEvidenceEvaluator,
    load_candidate_evidence_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Pareto candidate OOS evidence")
    parser.add_argument("--config", default="configs/studio_candidate_evidence.yaml")
    args = parser.parse_args()
    result = CandidateEvidenceEvaluator(
        load_candidate_evidence_config(args.config)
    ).run()
    print(f"Wrote {len(result.candidates)} candidates to {result.output_dir}")


if __name__ == "__main__":
    main()
