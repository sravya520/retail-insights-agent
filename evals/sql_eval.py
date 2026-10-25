"""Phase 2 SQL eval: execution accuracy against gold answers (numeric tolerance 0.5%).

Run: python evals/sql_eval.py
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.sql_agent import SQLAgent  # noqa: E402
from ria.sql_compare import results_match  # noqa: E402

EXAMPLE_IDS = ["q02", "q08", "q15"]  # fixed in advance, shown whether they pass or not


def main() -> None:
    config.LLM_CACHE_ENABLED = "--cache" in sys.argv  # final evals run with the cache OFF
    questions = [json.loads(l) for l in (ROOT / "evals" / "sql_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    agent = SQLAgent()
    records = []
    for q in questions:
        start = time.perf_counter()
        res = agent.run(q["question"])
        elapsed = time.perf_counter() - start
        ok = res.ok and results_match(q["gold_answer"], res.rows, ordered=q["ordered"])
        records.append({"q": q, "res": res, "ok": ok, "seconds": elapsed})
        print(f"{q['id']} {'PASS' if ok else 'FAIL'} {elapsed:5.1f}s attempts={res.attempts} {res.error or ''}", flush=True)

    n_ok = sum(r["ok"] for r in records)
    n_first = sum(r["ok"] and not r["res"].repaired for r in records)
    n_repaired_ok = sum(r["ok"] and r["res"].repaired for r in records)
    n_error = sum(not r["res"].ok for r in records)
    secs = [r["seconds"] for r in records]

    table = "\n".join(
        f"| {r['q']['id']} | {r['q']['question']} | {'pass' if r['ok'] else 'FAIL'} | "
        f"{'yes' if r['res'].repaired else 'no'} | {r['seconds']:.1f} |" for r in records)
    fails = [r for r in records if not r["ok"]]
    fail_text = "\n\n".join(
        f"**{r['q']['id']}** {r['q']['question']}\n- Gold SQL: `{r['q']['gold_sql']}`\n- Agent SQL: `{r['res'].sql}`\n"
        f"- Gold rows: {r['q']['gold_answer'][:5]}\n- Agent rows: {[list(x) for x in r['res'].rows[:5]]}\n"
        f"- Error: {r['res'].error}" for r in fails) or "None."
    examples = "\n\n".join(
        f"**Question:** {r['q']['question']}\n\n```sql\n{r['res'].sql}\n```\n\n**Answer:** {r['res'].answer}\n\n"
        f"Result matched gold: {'yes' if r['ok'] else 'NO'}"
        for r in records if r["q"]["id"] in EXAMPLE_IDS)

    update_section("phase2_sql", f"""### SQL agent execution accuracy ({len(records)} questions, all dev split)

Command: `python evals/sql_eval.py` (LLM cache off). Model: `{config.GEMINI_MODEL}`, thinking level for the SQL agent: `{config.THINKING_LEVELS.get('sql')}`.

- **Execution accuracy: {n_ok}/{len(records)} = {n_ok / len(records):.1%}** (agent result matches gold result, numeric tolerance 0.5%).
- Correct on the first query: {n_first}. Correct after the one repair: {n_repaired_ok}. Queries that failed even after repair: {n_error}.
- Time per question (writing, running and explaining): median {statistics.median(secs):.1f}s, max {max(secs):.1f}s.

| ID | Question | Result | Repaired | Seconds |
|---|---|---|---|---|
{table}

Failures in detail:

{fail_text}

### SQL example triples

{examples}""")


if __name__ == "__main__":
    main()
