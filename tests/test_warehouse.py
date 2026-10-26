"""BigQuery warehouse with a mocked client: transpiling, job limits, errors, schema, and the agent on top of it."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import duckdb
import pytest
import sqlglot
from google.api_core import exceptions as gexc

from ria import config
from ria.sql_agent import SQLAgent
from ria.warehouse import BigQueryWarehouse, DuckDBWarehouse, SQLExecutionError, default_warehouse, to_bigquery_sql

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = {"sales": ["invoice_no", "customer_id", "revenue", "invoice_year", "invoice_month", "country"],
          "returns": ["invoice_no", "quantity", "invoice_year"]}


class FakeRow(tuple):
    def values(self):
        return tuple(self)


class FakeJob:
    def __init__(self, columns, rows, bytes_billed=10_485_760, error=None):
        self.columns, self.rows, self.total_bytes_billed, self.error = columns, rows, bytes_billed, error

    def result(self, timeout=None):
        if self.error:
            raise self.error
        return _Result(self.columns, self.rows)


class _Result:
    def __init__(self, columns, rows):
        self.schema = [SimpleNamespace(name=c) for c in columns]
        self._rows = rows

    def __iter__(self):
        return iter(FakeRow(r) for r in self._rows)


class FakeClient:
    def __init__(self, job=None, tables=None):
        self.job, self.tables, self.queries = job, tables or SCHEMA, []

    def query(self, sql, job_config=None):
        self.queries.append((sql, job_config))
        return self.job

    def list_tables(self, dataset):
        return [SimpleNamespace(reference=name) for name in self.tables]

    def get_table(self, ref):
        return SimpleNamespace(table_id=ref, schema=[SimpleNamespace(name=c) for c in self.tables[ref]])


def make_bq(job=None):
    return BigQueryWarehouse(client=FakeClient(job), project="my-proj-123", dataset="retail",
                             max_bytes_billed=1000, timeout=7.0)


def test_transpile_qualifies_tables_but_not_ctes():
    sql = ("WITH t AS (SELECT customer_id, SUM(revenue) AS spend FROM sales GROUP BY customer_id) "
           "SELECT * FROM t JOIN returns r ON TRUE LIMIT 5")
    out = to_bigquery_sql(sql, "my-proj-123", "retail")
    assert "`my-proj-123`.`retail`.`sales`" in out
    assert "`my-proj-123`.`retail`.`returns`" in out
    assert "`my-proj-123`.`retail`.`t`" not in out
    sqlglot.parse_one(out, read="bigquery")  # valid BigQuery SQL


def test_transpile_duckdb_integer_division():
    out = to_bigquery_sql("SELECT (invoice_month - 1) // 3 + 1 AS quarter FROM sales", "p", "d")
    assert "DIV(" in out


def test_run_sets_bytes_cap_and_timeout_and_returns_rows():
    wh = make_bq(FakeJob(["country", "revenue"], [("Spain", 41494.97)], bytes_billed=20_971_520))
    columns, rows = wh.run("SELECT country, SUM(revenue) AS revenue FROM sales GROUP BY country LIMIT 200")
    assert columns == ["country", "revenue"] and rows == [("Spain", 41494.97)]
    sql, job_config = wh.client.queries[0]
    assert "`p" not in sql and "`my-proj-123`.`retail`.`sales`" in sql
    assert job_config.maximum_bytes_billed == 1000
    assert int(job_config.job_timeout_ms) == 7000  # the library stores it as a string (API format)
    assert wh.last_bytes_billed == 20_971_520 and wh.last_sql == sql


def test_bytes_cap_error_becomes_execution_error():
    err = gexc.BadRequest("Query exceeded limit for bytes billed: 1000. 10485760 or higher required.")
    wh = make_bq(FakeJob([], [], error=err))
    with pytest.raises(SQLExecutionError, match="bytes billed"):
        wh.run("SELECT country FROM sales LIMIT 200")


def test_timeout_becomes_execution_error():
    wh = make_bq(FakeJob([], [], error=TimeoutError()))
    with pytest.raises(SQLExecutionError, match="7 seconds"):
        wh.run("SELECT country FROM sales LIMIT 200")


def test_schema_comes_from_table_metadata():
    assert make_bq().schema() == SCHEMA


def test_agent_on_bigquery_validates_before_running():
    class ScriptedLLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def invoke(self, messages):
            return SimpleNamespace(content=self.replies.pop(0), usage_metadata=None, response_metadata={})

    # A DROP is blocked by the validator twice (write + repair), so BigQuery never sees a query.
    wh = make_bq(FakeJob(["n"], [(1,)]))
    llm = ScriptedLLM(["DROP TABLE sales", "DELETE FROM sales"])
    res = SQLAgent(llm=llm, warehouse=wh).run("Delete everything")
    assert not res.ok and wh.client.queries == []
    assert res.warehouse == "bigquery"


def test_agent_records_executed_sql_and_bytes(monkeypatch):
    monkeypatch.setattr(config, "LLM_CACHE_ENABLED", False)

    class ScriptedLLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def invoke(self, messages):
            return SimpleNamespace(content=self.replies.pop(0), usage_metadata=None, response_metadata={})

    wh = make_bq(FakeJob(["revenue"], [(41494.97,)], bytes_billed=10_485_760))
    llm = ScriptedLLM(["SELECT SUM(revenue) AS revenue FROM sales WHERE country = 'Spain' AND invoice_year = 2010",
                       "Revenue from Spain in 2010 was £41,494.97."])
    res = SQLAgent(llm=llm, warehouse=wh).run("Revenue from Spain in 2010?")
    assert res.ok and res.rows == [(41494.97,)]
    assert res.sql.startswith("SELECT SUM(revenue)")          # the validated DuckDB SQL (shown in the UI)
    assert "`my-proj-123`.`retail`.`sales`" in res.executed_sql  # what BigQuery ran
    assert res.bytes_billed == 10_485_760


def test_default_warehouse(monkeypatch):
    con = duckdb.connect()
    assert isinstance(default_warehouse(con), DuckDBWarehouse)
    monkeypatch.setattr(config, "WAREHOUSE", "snowflake")
    with pytest.raises(ValueError, match="duckdb or bigquery"):
        default_warehouse()


def test_bigquery_views_have_the_duckdb_view_columns():
    spec = importlib.util.spec_from_file_location("load_bigquery", ROOT / "scripts" / "load_bigquery.py")
    loader = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loader)
    con = duckdb.connect()
    con.execute("CREATE TABLE sales (invoice_no VARCHAR, stock_code VARCHAR, description VARCHAR, quantity BIGINT, "
                "invoice_date TIMESTAMP, unit_price DOUBLE, revenue DOUBLE, customer_id BIGINT, country VARCHAR, "
                "is_merchandise BOOLEAN)")
    from ria.sql_db import create_views
    create_views(con)
    for name, query in loader.view_queries("p", "d").items():
        expected = [r[0] for r in con.execute(f"DESCRIBE {name}").fetchall()]
        got = [e.alias_or_name for e in sqlglot.parse_one(query, read="bigquery").expressions]
        assert got == expected, name
