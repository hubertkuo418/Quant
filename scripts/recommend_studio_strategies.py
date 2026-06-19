from __future__ import annotations

import argparse

import pandas as pd

from equity_transformer.studio.recommendation import (
    load_recommendation_profile,
    recommend_strategies,
    save_recommendations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank Strategy Studio candidates for a user profile."
    )
    parser.add_argument("--profile", default="configs/studio_profile.yaml")
    parser.add_argument(
        "--candidates",
        default="artifacts/studio/optimizations/factor_search/results.csv",
    )
    parser.add_argument("--output-dir", default="artifacts/studio/recommendations")
    args = parser.parse_args()
    profile = load_recommendation_profile(args.profile)
    recommendations = recommend_strategies(pd.read_csv(args.candidates), profile)
    save_recommendations(recommendations, profile, args.output_dir)
    print(recommendations.to_string(index=False))


if __name__ == "__main__":
    main()
