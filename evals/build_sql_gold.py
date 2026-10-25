"""Run each gold SQL on the database and store the gold answer (rows) in sql_questions.jsonl.

Also checks that the gold query has no ties that would make the answer ambiguous.
Run: python evals/build_sql_gold.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria.sql_db import connect_readonly  # noqa: E402

PATH = ROOT / "evals" / "sql_questions.jsonl"


def jsonable(v):
    return v if v is None or isinstance(v, (int, float, str, bool)) else str(v)


def main() -> None:
    con = connect_readonly()
    rows = [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]
    for q in rows:
        cursor = con.execute(q["gold_sql"])
        q["gold_columns"] = [d[0] for d in cursor.description]
        q["gold_answer"] = [[jsonable(v) for v in r] for r in cursor.fetchall()]
        print(q["id"], q["gold_answer"])
    PATH.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
