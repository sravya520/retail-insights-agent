"""Where the SQL agent's validated queries run: local DuckDB (default) or Google BigQuery.

The model always writes DuckDB SQL and `validate_sql` always checks it in the DuckDB dialect. The BigQuery
warehouse then transpiles the safe query with sqlglot and qualifies every table with project and dataset.
So the prompt, the guardrails and the eval questions are identical on both warehouses.

BigQuery protections (on top of the validator):
  * A read-only service account (BigQuery Data Viewer + BigQuery Job User).
  * maximum_bytes_billed on every job: BigQuery rejects a query before it runs if it would bill more.
  * A job timeout. Query cache stays on (cached results bill 0 bytes).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from typing import Any, Protocol

import duckdb
import sqlglot
from sqlglot import exp

from ria import config


class SQLExecutionError(RuntimeError):
    """The SQL was allowed but failed or timed out."""


class Warehouse(Protocol):
    name: str

    def schema(self) -> dict[str, list[str]]: ...

    def run(self, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]: ...


class DuckDBWarehouse:
    name = "duckdb"

    def __init__(self, con: duckdb.DuckDBPyConnection, timeout: float = 5.0) -> None:
        self.con = con
        self.timeout = timeout
        self.last_sql: str = ""
        self.last_bytes_billed: int | None = None

    def schema(self) -> dict[str, list[str]]:
        from ria.sql_db import get_schema
        return get_schema(self.con)

    def run(self, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        """Run a validated query with a hard timeout. Interrupts the query when time is up."""
        self.last_sql = sql
        timer = threading.Timer(self.timeout, self.con.interrupt)
        timer.start()
        try:
            cursor = self.con.execute(sql)
            columns = [d[0] for d in cursor.description]
            return columns, cursor.fetchall()
        except duckdb.InterruptException as exc:
            raise SQLExecutionError(f"Query took longer than {self.timeout:g} seconds and was stopped.") from exc
        except duckdb.Error as exc:
            raise SQLExecutionError(str(exc)[:300]) from exc
        finally:
            timer.cancel()


def to_bigquery_sql(sql: str, project: str, dataset: str) -> str:
    """Transpile validated DuckDB SQL to BigQuery SQL and qualify the base tables (CTE names stay as they are)."""
    tree = sqlglot.parse_one(sql, read="duckdb")
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.name.lower() not in ctes:
            table.set("catalog", exp.to_identifier(project))
            table.set("db", exp.to_identifier(dataset))
    return tree.sql(dialect="bigquery", identify=True)


def bigquery_client(key_path: str, project: str, location: str) -> Any:
    """A BigQuery client from a service account key file (path from .env, the file lives outside the repo)."""
    from google.cloud import bigquery
    from google.oauth2 import service_account
    if not key_path:
        raise RuntimeError("No BigQuery key path. Set RIA_BQ_AGENT_KEY (or RIA_BQ_LOADER_KEY) in .env.")
    if not project:
        raise RuntimeError("No BigQuery project. Set RIA_BQ_PROJECT in .env.")
    creds = service_account.Credentials.from_service_account_file(
        key_path, scopes=["https://www.googleapis.com/auth/bigquery"])
    return bigquery.Client(project=project, credentials=creds, location=location)


@dataclass
class BigQueryWarehouse:
    client: Any
    project: str
    dataset: str
    max_bytes_billed: int = config.BQ_MAX_BYTES_BILLED
    timeout: float = config.BQ_TIMEOUT_SECONDS
    use_query_cache: bool = True  # cached results bill 0 bytes; measurements turn it off
    name: str = "bigquery"
    last_sql: str = ""
    last_bytes_billed: int | None = None
    _schema: dict[str, list[str]] | None = field(default=None, repr=False)

    @classmethod
    def from_config(cls) -> "BigQueryWarehouse":
        client = bigquery_client(config.BQ_AGENT_KEY, config.BQ_PROJECT, config.BQ_LOCATION)
        return cls(client=client, project=config.BQ_PROJECT, dataset=config.BQ_DATASET)

    def schema(self) -> dict[str, list[str]]:
        """Table and view columns from the table metadata API (metadata calls bill no query bytes)."""
        if self._schema is None:
            schema: dict[str, list[str]] = {}
            for item in self.client.list_tables(f"{self.project}.{self.dataset}"):
                table = self.client.get_table(item.reference)
                schema[table.table_id] = [f.name for f in table.schema]
            if not schema:
                raise RuntimeError(f"Dataset {self.project}.{self.dataset} has no tables. Run scripts/load_bigquery.py.")
            self._schema = schema
        return self._schema

    def run(self, sql: str) -> tuple[list[str], list[tuple[Any, ...]]]:
        from google.api_core import exceptions as gexc
        from google.cloud import bigquery
        bq_sql = to_bigquery_sql(sql, self.project, self.dataset)
        self.last_sql, self.last_bytes_billed = bq_sql, None
        job_config = bigquery.QueryJobConfig(maximum_bytes_billed=self.max_bytes_billed,
                                             job_timeout_ms=int(self.timeout * 1000),
                                             use_query_cache=self.use_query_cache)
        try:
            job = self.client.query(bq_sql, job_config=job_config)
            result = job.result(timeout=self.timeout)
            columns = [f.name for f in result.schema]
            rows = [tuple(row.values()) for row in result]
        except gexc.GoogleAPICallError as exc:
            raise SQLExecutionError(f"BigQuery error: {str(exc)[:300]}") from exc
        except TimeoutError as exc:
            raise SQLExecutionError(f"Query took longer than {self.timeout:g} seconds and was stopped.") from exc
        self.last_bytes_billed = job.total_bytes_billed
        return columns, rows


def default_warehouse(con: duckdb.DuckDBPyConnection | None = None) -> Warehouse:
    """The warehouse chosen by RIA_WAREHOUSE. A DuckDB connection passed in always wins (tests use one)."""
    if con is not None:
        return DuckDBWarehouse(con)
    if config.WAREHOUSE == "bigquery":
        return BigQueryWarehouse.from_config()
    if config.WAREHOUSE != "duckdb":
        raise ValueError(f"RIA_WAREHOUSE must be duckdb or bigquery, got {config.WAREHOUSE!r}")
    from ria.sql_db import connect_readonly
    return DuckDBWarehouse(connect_readonly())
