from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import yaml


@dataclass(frozen=True)
class CatalogConfig:
    database_path: Path
    manifest_path: Path
    tables: dict[str, Path]


def load_catalog_config(path: str | Path) -> CatalogConfig:
    with Path(path).open(encoding="utf-8") as stream:
        payload: dict[str, Any] = yaml.safe_load(stream)
    return CatalogConfig(
        database_path=Path(payload["database_path"]),
        manifest_path=Path(payload["manifest_path"]),
        tables={
            name: Path(table_path) for name, table_path in payload["tables"].items()
        },
    )


class DuckDBCatalogBuilder:
    def __init__(self, config: CatalogConfig) -> None:
        self.config = config

    def run(self) -> dict[str, object]:
        self.config.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        registered = []
        missing = []
        with duckdb.connect(str(self.config.database_path)) as connection:
            for table_name, parquet_path in self.config.tables.items():
                if parquet_path.exists():
                    safe_name = _safe_identifier(table_name)
                    path_literal = str(parquet_path).replace("\\", "/")
                    connection.execute(f"DROP VIEW IF EXISTS {safe_name}")
                    connection.execute(
                        f"CREATE VIEW {safe_name} AS "
                        f"SELECT * FROM read_parquet({_sql_literal(path_literal)})"
                    )
                    rows = connection.execute(
                        f"SELECT COUNT(*) FROM {safe_name}"
                    ).fetchone()[0]
                    registered.append(
                        {
                            "table": safe_name,
                            "path": str(parquet_path),
                            "rows": int(rows),
                        }
                    )
                else:
                    missing.append({"table": table_name, "path": str(parquet_path)})
        manifest = {
            "run_utc": datetime.now(UTC).isoformat(),
            "database_path": str(self.config.database_path),
            "registered": registered,
            "missing": missing,
        }
        self.config.manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        return manifest


def _safe_identifier(name: str) -> str:
    if not name.replace("_", "").isalnum() or name[0].isdigit():
        raise ValueError(f"Unsafe DuckDB identifier: {name}")
    return name


def _sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
