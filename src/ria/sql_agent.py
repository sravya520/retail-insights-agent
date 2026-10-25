"""Text-to-SQL agent. The LLM writes the SQL. Code decides whether it may run.

Guardrails (all in code, none in the prompt):
  * sqlglot parses the query. Exactly one SELECT statement is allowed (WITH ... SELECT is fine).
  * Only known tables and columns are allowed. File reading and system functions are blocked.
  * LIMIT 200 is added when missing, and any larger LIMIT is lowered to 200.
  * 5 second timeout, read-only connection.
  * On a validation or execution error the error goes back to the LLM once for a repair. Then we fail clearly.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

from ria import config
from ria.llm import LLMError, get_llm, invoke_with_retry
from ria.sql_db import SCHEMA_DESCRIPTION, connect_readonly, get_schema

MAX_ROWS = 200
TIMEOUT_SECONDS = 5.0
ROWS_SHOWN_TO_LLM = 30

# Never allowed anywhere in a query: statements that change data or settings.
FORBIDDEN_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
                   exp.Copy, exp.Merge, exp.Pragma, exp.Set, exp.Use, exp.Attach, exp.Detach,
                   exp.Transaction, exp.Commit, exp.Rollback, exp.Show, exp.Describe)
# Functions that read files or the system. Matched by name prefix, case-insensitive.
FORBIDDEN_FUNCTION_PREFIXES = ("read_", "glob", "sniff_", "query", "getenv", "current_setting", "duckdb_",
                               "pragma_", "parquet_", "json_read", "write_", "load_", "install_",
                               "list_files", "which_secret", "http_", "iceberg_", "delta_scan")


class SQLValidationError(ValueError):
    """The SQL is not allowed to run."""


class SQLExecutionError(RuntimeError):
    """The SQL was allowed but failed or timed out."""


def _cte_names(tree: exp.Expression) -> set[str]:
    return {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}


def _defined_aliases(tree: exp.Expression) -> set[str]:
    """Names the query itself defines: SELECT aliases and CTE or subquery column aliases."""
    names = {a.alias.lower() for a in tree.find_all(exp.Alias) if a.alias}
    for alias in tree.find_all(exp.TableAlias):
        names.update(c.name.lower() for c in alias.columns)
    return names


def validate_sql(sql: str, schema: dict[str, list[str]], max_rows: int = MAX_ROWS) -> str:
    """Return a safe, normalised version of `sql` (with LIMIT) or raise SQLValidationError."""
    if not sql or not sql.strip():
        raise SQLValidationError("Empty query.")
    try:
        statements = [s for s in sqlglot.parse(sql, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise SQLValidationError(f"SQL could not be parsed: {str(exc)[:200]}") from exc
    if len(statements) != 1:
        raise SQLValidationError(f"Only one statement is allowed, got {len(statements)}.")
    tree = statements[0]
    if not isinstance(tree, exp.Select):
        raise SQLValidationError(f"Only a single SELECT is allowed, got {type(tree).__name__.upper()}.")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SQLValidationError(f"{type(node).__name__.upper()} is not allowed.")
        if isinstance(node, exp.Func):
            name = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if name.startswith(FORBIDDEN_FUNCTION_PREFIXES):
                raise SQLValidationError(f"Function {name} is not allowed.")

    # The top-level SELECT list may not contain constants: a made-up value like "28 AS window_days" would look like data.
    for projection in tree.expressions:
        inner = projection.this if isinstance(projection, exp.Alias) else projection
        if isinstance(inner, exp.Literal):
            raise SQLValidationError("Constant values in the SELECT list are not allowed. Select columns or aggregates from the tables.")

    tables = {t.lower(): cols for t, cols in schema.items()}
    column_union = {c.lower() for cols in schema.values() for c in cols}
    ctes = _cte_names(tree)
    alias_to_table: dict[str, str] = {}
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLValidationError("Table functions are not allowed.")
        name = table.name.lower()
        if name in ctes:
            continue
        if name not in tables:
            raise SQLValidationError(f"Unknown table: {table.name}. Known tables: {', '.join(sorted(tables))}.")
        alias_to_table[(table.alias or table.name).lower()] = name
    defined = _defined_aliases(tree)
    for column in tree.find_all(exp.Column):
        col = column.name.lower()
        qualifier = column.table.lower() if column.table else ""
        if qualifier in alias_to_table:
            if col not in {c.lower() for c in tables[alias_to_table[qualifier]]} and col not in defined:
                raise SQLValidationError(f"Unknown column: {qualifier}.{column.name}")
        elif col not in column_union and col not in defined:
            raise SQLValidationError(f"Unknown column: {column.name}")

    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_rows)
    else:
        value = limit.expression
        if isinstance(value, exp.Literal) and value.is_int and int(value.name) > max_rows:
            tree = tree.limit(max_rows)
    return tree.sql(dialect="duckdb")


def execute_sql(con: duckdb.DuckDBPyConnection, sql: str,
                timeout: float = TIMEOUT_SECONDS) -> tuple[list[str], list[tuple[Any, ...]]]:
    """Run a validated query with a hard timeout. Interrupts the query when time is up."""
    timer = threading.Timer(timeout, con.interrupt)
    timer.start()
    try:
        cursor = con.execute(sql)
        columns = [d[0] for d in cursor.description]
        return columns, cursor.fetchall()
    except duckdb.InterruptException as exc:
        raise SQLExecutionError(f"Query took longer than {timeout:g} seconds and was stopped.") from exc
    except duckdb.Error as exc:
        raise SQLExecutionError(str(exc)[:300]) from exc
    finally:
        timer.cancel()


FEW_SHOT = """\
Example 1
Question: What were the top 3 countries by number of orders in 2010?
SQL:
SELECT country, COUNT(DISTINCT invoice_no) AS orders
FROM sales
WHERE invoice_year = 2010
GROUP BY country
ORDER BY orders DESC
LIMIT 3

Example 2
Question: What was the revenue in the first quarter of 2011?
SQL:
SELECT SUM(revenue) AS revenue_q1_2011
FROM sales
WHERE invoice_year = 2011 AND invoice_month BETWEEN 1 AND 3

Example 3
Question: How many different products were returned in 2011, and what was their total value?
SQL:
SELECT COUNT(DISTINCT stock_code) AS products_returned, SUM(revenue) AS value_returned
FROM returns
WHERE invoice_year = 2011 AND is_merchandise
"""

SYSTEM_PROMPT = f"""You write DuckDB SQL for a retail sales database. Reply with ONE SQL query and nothing else.

{SCHEMA_DESCRIPTION}
Rules:
  * Write a single read-only SELECT. Use only the tables and columns above.
  * Do not use semicolons or comments.
  * Return the columns needed to answer the question (a name next to a number is helpful).
  * If a question has no year, use all the data.

{FEW_SHOT}"""

REPAIR_TEMPLATE = """The previous query failed.
SQL:
{sql}
Error: {error}
Write a corrected single SELECT query. Reply with the SQL only."""

ANSWER_PROMPT = """You explain database results to a business user in plain English.
Use ONLY the numbers in the result table. Do not calculate new numbers and do not add facts.
Currency is GBP (£). Use 1 to 3 short sentences. If the table is empty, say that no data matched.

Question: {question}
SQL used: {sql}
Result ({total} rows{shown_note}):
{table}"""


@dataclass
class SQLResult:
    question: str
    sql: str = ""
    columns: list[str] = field(default_factory=list)
    rows: list[tuple[Any, ...]] = field(default_factory=list)
    answer: str = ""
    attempts: int = 0
    error: str | None = None
    repaired: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


def extract_sql(text: str) -> str:
    """Pull the SQL out of a model reply (handles ```sql fences and a trailing semicolon)."""
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, flags=re.S | re.I)
    sql = (fenced.group(1) if fenced else text).strip()
    return sql.rstrip(";").strip()


def format_table(columns: list[str], rows: list[tuple[Any, ...]], limit: int = ROWS_SHOWN_TO_LLM) -> str:
    def fmt(v: Any) -> str:
        return f"{v:,.2f}" if isinstance(v, float) else str(v)
    lines = [" | ".join(columns)] + [" | ".join(fmt(v) for v in row) for row in rows[:limit]]
    return "\n".join(lines)


def _text(reply: Any) -> str:
    content = reply.content
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)


class SQLAgent:
    def __init__(self, con: duckdb.DuckDBPyConnection | None = None, llm: Any = None) -> None:
        self.con = con or connect_readonly()
        self.schema = get_schema(self.con)
        self._llm = llm

    @property
    def llm(self) -> Any:
        if self._llm is None:
            self._llm = get_llm("sql")
        return self._llm

    def _generate(self, messages: list[tuple[str, str]], node: str) -> str:
        return extract_sql(_text(invoke_with_retry(self.llm, messages, node=node)))

    def run(self, question: str) -> SQLResult:
        result = SQLResult(question)
        messages = [("system", SYSTEM_PROMPT), ("human", f"Question: {question}\nSQL:")]
        try:
            sql = self._generate(messages, "sql_agent.write")
        except LLMError as exc:
            result.error = str(exc)
            result.answer = "I could not write a query because the language model call failed."
            return result
        for attempt in (1, 2):
            result.attempts = attempt
            result.sql = sql
            try:
                safe_sql = validate_sql(sql, self.schema)
                result.sql = safe_sql
                result.columns, result.rows = execute_sql(self.con, safe_sql)
                result.error = None
                break
            except (SQLValidationError, SQLExecutionError) as exc:
                result.error = str(exc)
                if attempt == 2:
                    break
                result.repaired = True
                repair = messages + [("ai", sql), ("human", REPAIR_TEMPLATE.format(sql=sql, error=exc))]
                try:
                    sql = self._generate(repair, "sql_agent.repair")
                except LLMError as llm_exc:
                    result.error = f"{exc} (repair failed: {llm_exc})"
                    break
        if result.error:
            result.answer = f"I could not answer this from the sales data. The query failed: {result.error}"
            return result
        result.answer = self._explain(result)
        return result

    def _explain(self, result: SQLResult) -> str:
        total = len(result.rows)
        shown_note = f", first {ROWS_SHOWN_TO_LLM} shown" if total > ROWS_SHOWN_TO_LLM else ""
        prompt = ANSWER_PROMPT.format(question=result.question, sql=result.sql, total=total,
                                      shown_note=shown_note, table=format_table(result.columns, result.rows))
        try:
            return _text(invoke_with_retry(self.llm, [("human", prompt)], node="sql_agent.explain")).strip()
        except LLMError as exc:
            return f"The query ran, but the explanation step failed: {exc}"
