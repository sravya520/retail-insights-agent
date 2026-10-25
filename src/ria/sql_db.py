"""Load the UCI Online Retail II dataset into DuckDB and describe the schema for the SQL agent.

Cleaning rules (all counted in data/db/load_report.json):
  * Invoices starting with "C" are cancellations. They go to the `returns` table.
  * `sales` holds normal invoice lines with quantity > 0 and price > 0.
    Lines with price <= 0 (bad debt adjustments, free or zero-price lines) are dropped.
  * Rows with a missing Customer ID are KEPT in `sales` (revenue counts them). customer_id is NULL for those rows.
  * revenue = quantity x unit_price.
  * is_merchandise is TRUE for real products (stock code starts with 5 digits). It is FALSE for
    postage, fees, manual adjustments and similar codes (POST, DOT, M, BANK CHARGES, and so on).
  * Exact duplicate rows are NOT removed, because repeated lines can be real.

Run: python -m ria.sql_db
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd

from ria import config

XLSX_PATH = config.DATA_DIR / "raw" / "online_retail_ii" / "online_retail_II.xlsx"
DB_PATH = config.DB_DIR / "sales.duckdb"
REPORT_PATH = config.DB_DIR / "load_report.json"

RENAME = {"Invoice": "invoice_no", "StockCode": "stock_code", "Description": "description",
          "Quantity": "quantity", "InvoiceDate": "invoice_date", "Price": "unit_price",
          "Customer ID": "customer_id", "Country": "country"}

SCHEMA_DESCRIPTION = """\
Database: DuckDB. Company: a UK online gift retailer. Data covers 2009-12-01 to 2011-12-09.
Only 2010 is a full calendar year. 2009 has December only. 2011 ends on 9 December.

TABLE sales  (one row per invoice line of a normal sale)
  invoice_no       VARCHAR   invoice number (one invoice = one order)
  stock_code       VARCHAR   product code
  description      VARCHAR   product name on this line (the same stock_code can have slightly different names)
  quantity         BIGINT    units sold (always > 0)
  invoice_date     TIMESTAMP date and time of the invoice
  invoice_year     INTEGER   year of invoice_date
  invoice_month    INTEGER   month of invoice_date (1 to 12)
  unit_price       DOUBLE    price per unit in GBP (always > 0)
  revenue          DOUBLE    quantity * unit_price in GBP. Total revenue = SUM(revenue)
  customer_id      BIGINT    customer id, NULL when the customer is unknown (these rows still count as revenue)
  country          VARCHAR   customer country
  is_merchandise   BOOLEAN   TRUE for real products. FALSE for postage, fees and adjustments.
                             For questions about products, use is_merchandise = TRUE.

TABLE returns  (one row per invoice line of a cancelled or returned order; invoice numbers start with 'C')
  invoice_no, stock_code, description, invoice_date, invoice_year, invoice_month, unit_price, customer_id,
  country, is_merchandise: same meaning as in sales
  quantity         BIGINT    units returned (always > 0)
  revenue          DOUBLE    value returned in GBP = quantity * unit_price (always > 0)

VIEW products  (one row per stock_code, built from sales)
  stock_code, description (most common name), is_merchandise, avg_unit_price

VIEW customers  (one row per known customer_id, built from sales)
  customer_id, country, first_order_date, last_order_date, order_count, total_revenue

Rules of thumb:
  * Number of orders = COUNT(DISTINCT invoice_no).
  * Revenue figures come from sales. Returns are in the returns table and are not subtracted from sales.
  * To name a product, group by stock_code and use the name from the products view.
  * Use invoice_year and invoice_month for date filters.
"""


def clean_lines(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Split raw Online Retail II rows into clean `sales` and `returns` frames. Returns a count report."""
    df = raw.rename(columns=RENAME)[list(RENAME.values())].copy()
    report: dict[str, int] = {"raw_rows": len(df)}
    df["invoice_no"] = df["invoice_no"].astype(str).str.strip()
    df["stock_code"] = df["stock_code"].astype(str).str.strip().str.upper()
    df["description"] = df["description"].astype("string").str.strip()
    df["customer_id"] = pd.to_numeric(df["customer_id"], errors="coerce").astype("Int64")
    df["country"] = df["country"].astype(str).str.strip()
    df["invoice_date"] = pd.to_datetime(df["invoice_date"])
    df["invoice_year"] = df["invoice_date"].dt.year.astype("int32")
    df["invoice_month"] = df["invoice_date"].dt.month.astype("int32")
    df["is_merchandise"] = df["stock_code"].str.match(r"^\d{5}")

    cancelled = df["invoice_no"].str.startswith("C")
    report["cancellation_rows"] = int(cancelled.sum())
    sales = df[~cancelled & (df["quantity"] > 0) & (df["unit_price"] > 0)].copy()
    report["sales_rows_dropped_bad_quantity_or_price"] = int((~cancelled).sum() - len(sales))
    returns = df[cancelled & (df["quantity"] < 0) & (df["unit_price"] > 0)].copy()
    report["returns_rows_dropped_bad_quantity_or_price"] = int(cancelled.sum() - len(returns))
    returns["quantity"] = -returns["quantity"]
    for frame in (sales, returns):
        frame["revenue"] = frame["quantity"] * frame["unit_price"]
    report["sales_rows"] = len(sales)
    report["returns_rows"] = len(returns)
    report["sales_rows_without_customer_id"] = int(sales["customer_id"].isna().sum())
    return sales, returns, report


def build_db(xlsx_path: Path = XLSX_PATH, db_path: Path = DB_PATH) -> dict[str, Any]:
    sheets = pd.read_excel(xlsx_path, sheet_name=None, engine="openpyxl")
    raw = pd.concat(sheets.values(), ignore_index=True)
    sales, returns, report = clean_lines(raw)
    report["sheets"] = {name: len(frame) for name, frame in sheets.items()}

    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = duckdb.connect(str(db_path))
    cols = ("invoice_no, stock_code, description, quantity, invoice_date, invoice_year, invoice_month, "
            "unit_price, revenue, customer_id, country, is_merchandise")
    con.register("sales_df", sales)
    con.register("returns_df", returns)
    con.execute(f"CREATE TABLE sales AS SELECT {cols} FROM sales_df")
    con.execute(f"CREATE TABLE returns AS SELECT {cols} FROM returns_df")
    con.execute("""CREATE VIEW products AS
        SELECT stock_code, mode(description) AS description, bool_or(is_merchandise) AS is_merchandise,
               avg(unit_price) AS avg_unit_price FROM sales GROUP BY stock_code""")
    con.execute("""CREATE VIEW customers AS
        SELECT customer_id, mode(country) AS country, min(invoice_date) AS first_order_date,
               max(invoice_date) AS last_order_date, count(DISTINCT invoice_no) AS order_count,
               sum(revenue) AS total_revenue FROM sales WHERE customer_id IS NOT NULL GROUP BY customer_id""")
    report["date_min"] = str(con.execute("SELECT min(invoice_date) FROM sales").fetchone()[0])
    report["date_max"] = str(con.execute("SELECT max(invoice_date) FROM sales").fetchone()[0])
    report["total_revenue_gbp"] = round(con.execute("SELECT sum(revenue) FROM sales").fetchone()[0], 2)
    con.close()
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def connect_readonly(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    """Read-only connection with file access from SQL switched off (defence in depth, the validator comes first)."""
    return duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})


def get_schema(con: duckdb.DuckDBPyConnection) -> dict[str, list[str]]:
    """Map every table and view name to its column names."""
    rows = con.execute(
        "SELECT table_name, column_name FROM information_schema.columns "
        "WHERE table_schema = 'main' ORDER BY table_name, ordinal_position").fetchall()
    schema: dict[str, list[str]] = {}
    for table, column in rows:
        schema.setdefault(table, []).append(column)
    return schema


def main() -> None:
    report = build_db()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
