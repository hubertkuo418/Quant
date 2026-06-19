from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from equity_transformer.data.catalog import (
    CatalogConfig,
    DuckDBCatalogBuilder,
    _safe_identifier,
)


def test_duckdb_catalog_registers_existing_parquet_and_missing_files(
    tmp_path: Path,
) -> None:
    parquet_path = tmp_path / "panel.parquet"
    pd.DataFrame({"date": ["2024-01-01"], "ticker": ["AAA"], "value": [1]}).to_parquet(
        parquet_path,
        index=False,
    )
    config = CatalogConfig(
        database_path=tmp_path / "catalog.duckdb",
        manifest_path=tmp_path / "manifest.json",
        tables={
            "factor_panel": parquet_path,
            "missing_panel": tmp_path / "missing.parquet",
        },
    )

    manifest = DuckDBCatalogBuilder(config).run()

    assert manifest["registered"][0]["table"] == "factor_panel"
    assert manifest["registered"][0]["rows"] == 1
    assert manifest["missing"][0]["table"] == "missing_panel"
    with duckdb.connect(str(config.database_path)) as connection:
        rows = connection.execute("SELECT COUNT(*) FROM factor_panel").fetchone()[0]
    assert rows == 1
    assert config.manifest_path.exists()


def test_duckdb_catalog_rejects_unsafe_identifier() -> None:
    with pytest.raises(ValueError, match="Unsafe"):
        _safe_identifier("bad-name")
