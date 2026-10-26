"""Warehouse parity: run the same SQL on DuckDB and on BigQuery and compare each result to the gold answer.

No LLM calls. Two sets of queries:
  1. Gold SQL of every question that has one (eval v1, eval v2 and the Phase 2 SQL set, duplicates removed).
  2. Agent SQL replayed from saved run records (what the model actually wrote), whose question has a gold answer.
Every query goes through validate_sql first, exactly like the agent. BigQuery bytes billed are recorded per query.

Run: python scripts/bq_parity.py [--no-write]
Needs in .env: RIA_BQ_PROJECT and RIA_BQ_AGENT_KEY (read-only key), and data loaded with scripts/load_bigquery.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.sql_agent import SQLValidationError, validate_sql  # noqa: E402
from ria.sql_compare import results_match  # noqa: E402
from ria.sql_db import connect_readonly  # noqa: E402
from ria.warehouse import BigQueryWarehouse, DuckDBWarehouse, SQLExecutionError  # noqa: E402

QUESTION_FILES = ["evals/questions.jsonl", "evals/questions_v2.jsonl", "evals/sql_questions.jsonl"]
RECORD_FILES = ["evals/runs/run-phase4/records.jsonl", "evals/runs/run-v2/records.jsonl"]


def read_jsonl(rel: str) -> list[dict]:
    path = ROOT / rel
    return [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()] if path.exists() else []


def cases() -> list[dict]:
    """(source, id, sql, gold, ordered) for every distinct query to check."""
    out, seen = [], set()
    for rel in QUESTION_FILES:
        for q in read_jsonl(rel):
            if q.get("gold_sql") and q["gold_sql"] not in seen:
                seen.add(q["gold_sql"])
                out.append({"source": "gold", "id": q["id"], "sql": q["gold_sql"], "gold": q["gold_answer"],
                            "ordered": q.get("ordered", False)})
    for rel in RECORD_FILES:
        for r in read_jsonl(rel):
            sql = (r.get("sql") or {}).get("sql")
            if sql and r.get("gold_answer") is not None and not (r["sql"] or {}).get("error") and sql not in seen:
                seen.add(sql)
                out.append({"source": f"agent ({Path(rel).parent.name})", "id": r["id"], "sql": sql,
                            "gold": r["gold_answer"], "ordered": r.get("ordered", False)})
    return out


def run_one(wh, safe_sql: str) -> tuple[list | None, str | None, float]:
    start = time.perf_counter()
    try:
        _, rows = wh.run(safe_sql)
        return [list(r) for r in rows], None, time.perf_counter() - start
    except SQLExecutionError as exc:
        return None, str(exc)[:200], time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-write", action="store_true", help="print only, do not update docs/results.md")
    args = parser.parse_args()

    duck = DuckDBWarehouse(connect_readonly())
    bq = BigQueryWarehouse.from_config()
    bq.use_query_cache = False  # measure real bytes billed and wall time, not cache hits
    schema = duck.schema()
    bq_schema = bq.schema()
    if {t: sorted(c) for t, c in schema.items()} != {t: sorted(c) for t, c in bq_schema.items()}:
        print("WARNING: schemas differ", schema, bq_schema)

    rows_out, results = [], []
    for c in cases():
        try:
            safe = validate_sql(c["sql"], schema)
        except SQLValidationError as exc:
            print(f"{c['id']} skipped (validation): {exc}")
            continue
        d_rows, d_err, d_s = run_one(duck, safe)
        b_rows, b_err, b_s = run_one(bq, safe)
        d_ok = d_err is None and results_match(c["gold"], d_rows, ordered=c["ordered"])
        b_ok = b_err is None and results_match(c["gold"], b_rows, ordered=c["ordered"])
        res = {**c, "duck_ok": d_ok, "bq_ok": b_ok, "duck_s": d_s, "bq_s": b_s, "bq_bytes": bq.last_bytes_billed,
               "bq_error": b_err, "bq_sql": bq.last_sql, "bq_rows": (b_rows or [])[:3], "duck_rows": (d_rows or [])[:3]}
        results.append(res)
        print(f"{c['id']:>4} {c['source']:<22} duckdb {'pass' if d_ok else 'FAIL'}  bigquery {'pass' if b_ok else 'FAIL'}"
              f"  {b_s:5.2f}s  {bq.last_bytes_billed}  {b_err or ''}", flush=True)
        rows_out.append(f"| {c['id']} | {c['source']} | {'pass' if d_ok else 'FAIL'} | {'pass' if b_ok else 'FAIL'} | "
                        f"{(res['bq_bytes'] or 0) / 1e6:.1f} | {d_s:.2f} | {b_s:.2f} |")

    # Live check of the bytes cap: the largest query above, under a cap smaller than what it needs, must be rejected.
    biggest = max(results, key=lambda r: r["bq_bytes"] or 0)
    small_cap = 20 * 1024 * 1024
    capped = BigQueryWarehouse(client=bq.client, project=bq.project, dataset=bq.dataset,
                               max_bytes_billed=small_cap, use_query_cache=False)
    try:
        capped.run(validate_sql(biggest["sql"], schema))
        cap_text = f"NOT rejected: {biggest['id']} ran under a cap of {small_cap:,} bytes."
    except SQLExecutionError as exc:
        cap_text = (f"rejected as expected. Query {biggest['id']} ({biggest['source']}, {biggest['bq_bytes']:,} bytes billed "
                    f"without the cap) under a cap of {small_cap:,} bytes. BigQuery said: \"{str(exc)[:160]}\"")
    print("bytes cap check:", cap_text)

    n = len(results)
    same = sum(r["duck_ok"] == r["bq_ok"] for r in results)
    billed = [r["bq_bytes"] or 0 for r in results]
    diffs = [r for r in results if r["duck_ok"] != r["bq_ok"] or r["bq_error"]]
    diff_text = "\n".join(
        f"- **{r['id']}** ({r['source']}): DuckDB {'pass' if r['duck_ok'] else 'FAIL'}, BigQuery "
        f"{'pass' if r['bq_ok'] else 'FAIL'}. Error: {r['bq_error'] or 'none'}. BigQuery rows: {r['bq_rows']}. "
        f"DuckDB rows: {r['duck_rows']}. BigQuery SQL: `{r['bq_sql']}`" for r in diffs) or "None."
    print(f"{n} queries, same outcome on both: {same}. DuckDB pass {sum(r['duck_ok'] for r in results)}, "
          f"BigQuery pass {sum(r['bq_ok'] for r in results)}. Bytes billed total {sum(billed)}")
    (ROOT / "logs").mkdir(exist_ok=True)
    (ROOT / "logs" / "bq_parity.json").write_text(json.dumps(results, indent=1, default=str), encoding="utf-8")
    if args.no_write:
        return
    update_section("cloud_bigquery_parity", f"""### BigQuery warehouse parity (same SQL on DuckDB and BigQuery)

Command: `python scripts/bq_parity.py`. No LLM calls. Each query is validated by `validate_sql` (DuckDB dialect), then run on
local DuckDB and on BigQuery (sandbox, project from `.env`, dataset `{config.BQ_DATASET}`, read-only service account). On BigQuery
the query is transpiled by sqlglot and every job has `maximum_bytes_billed` = {config.BQ_MAX_BYTES_BILLED:,} bytes. Each result is
compared with the gold answer (numeric tolerance 0.5%). Queries: gold SQL from the question files, plus agent SQL replayed from
`{'`, `'.join(RECORD_FILES)}` (only questions with a gold answer and no SQL error, duplicates removed). The BigQuery query cache
was off for this run, so the bytes and times are real (the agent keeps the cache on). BigQuery bills at least 10 MiB per table read.
MB below means 1,000,000 bytes.

- Queries checked: {n}. Same outcome on both warehouses: **{same} of {n}**.
- Matched gold: DuckDB {sum(r['duck_ok'] for r in results)} of {n}, BigQuery {sum(r['bq_ok'] for r in results)} of {n}.
  Failed on both: {', '.join(f"{r['id']} ({r['source']})" for r in results if not r['duck_ok'] and not r['bq_ok']) or 'none'}.
- Bytes cap check: {cap_text}
- BigQuery bytes billed: total {sum(billed) / 1e6:.1f} MB, median {statistics.median(billed) / 1e6:.1f} MB per query, max {max(billed) / 1e6:.1f} MB.
- Wall time per query: DuckDB median {statistics.median(r['duck_s'] for r in results):.2f} s, BigQuery median {statistics.median(r['bq_s'] for r in results):.2f} s.

Differences and BigQuery errors:

{diff_text}

| ID | Source | DuckDB | BigQuery | BigQuery MB billed | DuckDB s | BigQuery s |
|---|---|---|---|---|---|---|
{chr(10).join(rows_out)}
""")
    print("written to docs/results.md")


if __name__ == "__main__":
    main()
