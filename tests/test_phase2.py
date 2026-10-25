import duckdb
import pandas as pd
import pytest

from ria.sql_agent import (SQLAgent, SQLExecutionError, SQLValidationError, execute_sql, extract_sql,
                           validate_sql)
from ria.sql_db import clean_lines

SCHEMA = {
    "sales": ["invoice_no", "stock_code", "description", "quantity", "invoice_date", "invoice_year",
              "invoice_month", "unit_price", "revenue", "customer_id", "country", "is_merchandise"],
    "returns": ["invoice_no", "stock_code", "quantity", "revenue", "invoice_year", "country"],
    "products": ["stock_code", "description"],
}


@pytest.mark.parametrize("sql", [
    "DROP TABLE sales",
    "DELETE FROM sales",
    "UPDATE sales SET quantity = 1",
    "INSERT INTO sales VALUES (1)",
    "CREATE TABLE x AS SELECT 1",
    "SELECT 1; SELECT 2",
    "SELECT * FROM sales; DROP TABLE sales",
    "SELECT * FROM secrets",
    "SELECT * FROM read_csv('/etc/passwd')",
    "SELECT * FROM 'file.csv'",
    "SELECT getenv('HOME')",
    "SELECT nonexistent_col FROM sales",
    "SELECT s.nonexistent FROM sales s",
    "COPY sales TO 'x.csv'",
    "PRAGMA database_list",
    "ATTACH 'other.db'",
    "SELECT 1 UNION SELECT 2",
    "SELECT COUNT(*) AS n, 28 AS return_window_days FROM returns",   # made-up constant looks like data
    "SELECT 42",
    "",
    "not sql at all",
])
def test_guardrail_rejects(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql, SCHEMA)


@pytest.mark.parametrize("sql", [
    "SELECT SUM(revenue) FROM sales WHERE invoice_year = 2010",
    "SELECT country, COUNT(DISTINCT invoice_no) AS n FROM sales GROUP BY country ORDER BY n DESC",
    "WITH m AS (SELECT invoice_month, SUM(revenue) AS rev FROM sales GROUP BY invoice_month) "
    "SELECT invoice_month, rev FROM m ORDER BY rev DESC",
    "SELECT s.stock_code, p.description, SUM(s.revenue) AS r FROM sales s JOIN products p "
    "ON s.stock_code = p.stock_code GROUP BY s.stock_code, p.description",
    "SELECT EXTRACT(year FROM invoice_date) AS y, strftime(invoice_date, '%Y-%m') AS ym FROM sales",
    "SELECT * FROM sales",
    "SELECT SUM(revenue) FROM sales;",
])
def test_guardrail_accepts(sql):
    assert validate_sql(sql, SCHEMA).upper().startswith(("SELECT", "WITH"))


def test_limit_added_when_missing():
    assert validate_sql("SELECT country FROM sales", SCHEMA).upper().endswith("LIMIT 200")


def test_limit_kept_when_small_and_lowered_when_large():
    assert validate_sql("SELECT country FROM sales LIMIT 5", SCHEMA).upper().endswith("LIMIT 5")
    assert validate_sql("SELECT country FROM sales LIMIT 100000", SCHEMA).upper().endswith("LIMIT 200")


def test_timeout_stops_slow_query():
    con = duckdb.connect()
    slow = "SELECT count(*) FROM range(3000000000) a, range(3000000000) b"
    with pytest.raises(SQLExecutionError, match="stopped"):
        execute_sql(con, slow, timeout=0.5)


def test_read_only_connection_blocks_writes(tmp_path):
    path = tmp_path / "t.duckdb"
    duckdb.connect(str(path)).execute("CREATE TABLE sales AS SELECT 1 AS quantity")
    ro = duckdb.connect(str(path), read_only=True)
    with pytest.raises(duckdb.Error):
        ro.execute("DELETE FROM sales")


def test_extract_sql():
    assert extract_sql("```sql\nSELECT 1;\n```") == "SELECT 1"
    assert extract_sql("SELECT 2;") == "SELECT 2"


def test_clean_lines_rules():
    raw = pd.DataFrame({
        "Invoice": ["1001", "1002", "C1003", "1004", "A1005", "1006"],
        "StockCode": ["85123A", "POST", "85123A", "22111", "B", "20000"],
        "Description": ["HEART", "POSTAGE", "HEART", "BOX", "Adjust bad debt", "FREE"],
        "Quantity": [2, 1, -3, 5, 1, 4],
        "InvoiceDate": pd.to_datetime(["2010-12-01 10:00"] * 6),
        "Price": [2.5, 10.0, 2.5, 1.0, -50.0, 0.0],
        "Customer ID": [12345.0, None, 12345.0, None, None, 1.0],
        "Country": ["United Kingdom"] * 6,
    })
    sales, returns, report = clean_lines(raw)
    assert sorted(sales["invoice_no"]) == ["1001", "1002", "1004"]  # A-invoice and zero-price line dropped
    assert report["sales_rows_without_customer_id"] == 2  # rows with no customer id are kept
    assert list(returns["invoice_no"]) == ["C1003"]
    assert returns["quantity"].iloc[0] == 3 and returns["revenue"].iloc[0] == 7.5  # positive units and value
    assert sales.set_index("invoice_no").loc["1001", "revenue"] == 5.0
    merch = sales.set_index("invoice_no")["is_merchandise"]
    assert bool(merch["1001"]) is True and bool(merch["1002"]) is False  # POST is not merchandise


class _Reply:
    def __init__(self, content):
        self.content = content
        self.usage_metadata = {}


class _ScriptedLLM:
    """Returns scripted replies in order, to test the repair loop without the real model."""
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        return _Reply(self.replies.pop(0))


@pytest.fixture()
def tiny_con():
    con = duckdb.connect()
    con.execute("CREATE TABLE sales AS SELECT 'I1' AS invoice_no, 'A' AS stock_code, 'X' AS description, "
                "2 AS quantity, TIMESTAMP '2010-01-01' AS invoice_date, 2010 AS invoice_year, 1 AS invoice_month, "
                "3.0 AS unit_price, 6.0 AS revenue, 1 AS customer_id, 'UK' AS country, true AS is_merchandise")
    return con


def test_agent_repairs_once_then_answers(tiny_con):
    llm = _ScriptedLLM(["SELECT SUM(revenu) FROM sales", "SELECT SUM(revenue) AS total FROM sales",
                        "Total revenue is £6.00."])
    result = SQLAgent(con=tiny_con, llm=llm).run("What is total revenue?")
    assert result.ok and result.repaired and result.attempts == 2
    assert result.rows == [(6.0,)] and result.answer == "Total revenue is £6.00."


def test_agent_fails_clearly_after_one_repair(tiny_con):
    llm = _ScriptedLLM(["DROP TABLE sales", "DELETE FROM sales"])
    result = SQLAgent(con=tiny_con, llm=llm).run("Delete everything")
    assert not result.ok and result.attempts == 2 and llm.calls == 2
    assert "could not answer" in result.answer
    assert tiny_con.execute("SELECT count(*) FROM sales").fetchone()[0] == 1  # data untouched


from ria.sql_compare import cell_match, results_match  # noqa: E402


def test_numeric_tolerance_half_percent():
    assert cell_match(1000.0, 1004.9)      # 0.49% off: ok
    assert not cell_match(1000.0, 1006.0)  # 0.6% off: not ok
    assert cell_match(5, 5.0)


def test_results_match_allows_extra_columns_and_ignores_order_when_unordered():
    gold = [("A", 10.0), ("B", 20.0)]
    agent = [("B", 20.1, "extra"), ("A", 10.0, "extra")]
    assert results_match(gold, agent, ordered=False)
    assert not results_match(gold, agent, ordered=True)


def test_results_match_rejects_wrong_rowcount_and_values():
    assert not results_match([(1,)], [(1,), (2,)])
    assert not results_match([("A", 10.0)], [("A", 99.0)])
    assert results_match([("Rabbit Night Light", 5)], [("RABBIT NIGHT LIGHT ", 5)])
