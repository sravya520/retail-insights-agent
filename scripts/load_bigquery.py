"""Load the committed parquet files into BigQuery and create the products and customers views.

Uses the LOADER service account (BigQuery Data Editor + BigQuery Job User). The agent uses a separate read-only key.
Safe to run again: tables are replaced (WRITE_TRUNCATE) and views are re-created.
Sandbox note: tables and views expire after 60 days. Run this again to restore them.

Run: python scripts/load_bigquery.py
Needs in .env: RIA_BQ_PROJECT, RIA_BQ_LOADER_KEY (a path outside the repo). Optional: RIA_BQ_DATASET, RIA_BQ_LOCATION.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria.warehouse import bigquery_client  # noqa: E402

TABLES = ("sales", "returns")


def mode_subquery(source: str, value: str, key: str) -> str:
    """Most common non-null `value` per `key` (DuckDB mode()). Ties go to the smallest value, so the result is stable."""
    return f"""SELECT {key}, {value} FROM (
      SELECT {key}, {value}, ROW_NUMBER() OVER (PARTITION BY {key} ORDER BY COUNT(*) DESC, {value}) AS rn
      FROM {source} WHERE {value} IS NOT NULL GROUP BY {key}, {value}) WHERE rn = 1"""


def view_queries(project: str, dataset: str) -> dict[str, str]:
    """BigQuery versions of the views in ria.sql_db.create_views (same columns, same meaning)."""
    sales = f"`{project}.{dataset}.sales`"
    return {
        "products": f"""SELECT s.stock_code, d.description, s.is_merchandise, s.avg_unit_price
FROM (SELECT stock_code, LOGICAL_OR(is_merchandise) AS is_merchandise, AVG(unit_price) AS avg_unit_price
      FROM {sales} GROUP BY stock_code) AS s
LEFT JOIN ({mode_subquery(sales, 'description', 'stock_code')}) AS d USING (stock_code)""",
        "customers": f"""SELECT a.customer_id, c.country, a.first_order_date, a.last_order_date, a.order_count, a.total_revenue
FROM (SELECT customer_id, MIN(invoice_date) AS first_order_date, MAX(invoice_date) AS last_order_date,
             COUNT(DISTINCT invoice_no) AS order_count, SUM(revenue) AS total_revenue
      FROM {sales} WHERE customer_id IS NOT NULL GROUP BY customer_id) AS a
LEFT JOIN ({mode_subquery(sales, 'country', 'customer_id')}) AS c USING (customer_id)""",
    }


def load(client, project: str, dataset: str, location: str, parquet_dir: Path) -> dict[str, dict]:
    from google.cloud import bigquery
    ds = bigquery.Dataset(f"{project}.{dataset}")
    ds.location = location
    client.create_dataset(ds, exists_ok=True)
    report: dict[str, dict] = {}
    for table in TABLES:
        path = parquet_dir / f"{table}.parquet"
        job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.PARQUET,
                                            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
        with path.open("rb") as f:
            client.load_table_from_file(f, f"{project}.{dataset}.{table}", job_config=job_config).result()
        t = client.get_table(f"{project}.{dataset}.{table}")
        report[table] = {"rows": t.num_rows, "bytes": t.num_bytes,
                         "columns": {f.name: f.field_type for f in t.schema}}
    for name, query in view_queries(project, dataset).items():
        ref = f"{project}.{dataset}.{name}"
        client.delete_table(ref, not_found_ok=True)
        view = bigquery.Table(ref)
        view.view_query = query
        client.create_table(view)
        report[name] = {"view": True}
    return report


def main() -> None:
    client = bigquery_client(config.BQ_LOADER_KEY, config.BQ_PROJECT, config.BQ_LOCATION)
    report = load(client, config.BQ_PROJECT, config.BQ_DATASET, config.BQ_LOCATION, config.DEPLOY_DIR)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
