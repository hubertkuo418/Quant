from __future__ import annotations

import argparse

from equity_transformer.strategies.prediction_signals import (
    PredictionSignalConfig,
    convert_prediction_config,
    load_prediction_signal_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert model test predictions into strategy signal parquet."
    )
    parser.add_argument("--config", default="configs/prediction_signals.yaml")
    parser.add_argument("--input", help="Override prediction parquet path.")
    parser.add_argument("--output", help="Override signal parquet path.")
    parser.add_argument("--horizon", type=int, help="Override forecast horizon.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--score-column", default=None)
    parser.add_argument("--metadata", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = load_prediction_signal_config(args.config)
    config = PredictionSignalConfig(
        input_path=base.input_path if args.input is None else args.input,
        output_path=base.output_path if args.output is None else args.output,
        horizon=base.horizon if args.horizon is None else args.horizon,
        model=base.model if args.model is None else args.model,
        score_column=(
            base.score_column if args.score_column is None else args.score_column
        ),
        metadata_path=base.metadata_path if args.metadata is None else args.metadata,
    )
    signal = convert_prediction_config(config)
    print(
        f"Saved {len(signal):,} prediction signals across "
        f"{signal['date'].nunique() if not signal.empty else 0} dates."
    )


if __name__ == "__main__":
    main()
