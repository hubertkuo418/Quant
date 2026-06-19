from __future__ import annotations

import argparse

from equity_transformer.reporting.config import load_report_config
from equity_transformer.reporting.summary import ReportPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dashboard-ready reports.")
    parser.add_argument("--config", default="configs/reporting.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_report_config(parse_args().config)
    outputs = ReportPipeline(config).run()
    for name, frame in outputs.items():
        print(f"{name}: {len(frame):,} rows")


if __name__ == "__main__":
    main()
