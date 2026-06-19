# DuckDB Catalog

QuantLab can register generated Parquet artifacts as DuckDB views for fast local
SQL exploration.

## Workflow

```powershell
python scripts/build_catalog.py --config configs/catalog.yaml
```

Outputs:

```text
data/catalog.duckdb
data/metadata/catalog_manifest.json
```

The builder creates views for existing Parquet files and records missing files
in the manifest instead of failing the whole run. This keeps the catalog useful
while the research workflow is still partially populated.

Example:

```sql
SELECT date, ticker, alpha101_001
FROM factor_panel
WHERE alpha101_001 IS NOT NULL
LIMIT 20;
```
