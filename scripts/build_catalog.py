from __future__ import annotations

import argparse

from equity_transformer.data.catalog import DuckDBCatalogBuilder, load_catalog_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DuckDB catalog views.")
    parser.add_argument("--config", default="configs/catalog.yaml")
    return parser.parse_args()


def main() -> None:
    config = load_catalog_config(parse_args().config)
    manifest = DuckDBCatalogBuilder(config).run()
    print(
        f"Registered {len(manifest['registered'])} tables; "
        f"{len(manifest['missing'])} missing."
    )


if __name__ == "__main__":
    main()
