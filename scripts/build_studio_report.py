from __future__ import annotations

import argparse

from equity_transformer.studio.report import build_studio_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Strategy Studio report.")
    parser.add_argument(
        "--output", default="docs/STRATEGY_STUDIO_RESULTS.md"
    )
    args = parser.parse_args()
    report = build_studio_report(args.output)
    print(f"Wrote {len(report):,} characters to {args.output}")


if __name__ == "__main__":
    main()
