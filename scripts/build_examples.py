"""Build data/examples/examples_v1.json: pre-computed answers for the public demo (no quota needed to open them).

Source: the saved records of a REAL run of the full system on the 20 eval-v1 held-out questions
(evals/runs/caching-stable_first.jsonl, model gemini-3.5-flash-lite, run on 2026-10-25 as part of the Phase 5 experiment).
Nothing is written by hand: answers, SQL, citations, tokens, cost and latency are exactly what that run recorded.
Each example also says whether that run's answer matched the eval gold answer, so wrong answers are shown as wrong.

Run: python scripts/build_examples.py     (no API calls)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria.service import shape_answer  # noqa: E402
from ria.sql_compare import results_match  # noqa: E402

SOURCE = ROOT / "evals" / "runs" / "caching-stable_first.jsonl"
OUT = config.EXAMPLES_PATH
RUN_NOTE = "A real run of the full system as of git tag eval-v1 (before the later router and prompt fixes) on an eval-v1 held-out question (run caching-stable_first, gemini-3.5-flash-lite, 2026-10-25)."


def matches_gold(rec: dict) -> bool | None:
    """Data and both: did the SQL result match the gold result? Policy: was a gold clause retrieved? Else None."""
    if rec.get("gold_answer") is not None:
        sql = rec.get("sql")
        return bool(sql and not sql.get("error") and results_match(rec["gold_answer"], sql["rows"], ordered=rec.get("ordered", False)))
    if rec.get("gold_citations"):
        gold = {(g["doc"], g["clause_id"]) for g in rec["gold_citations"]}
        got = {(c["doc"], c["clause_id"]) for c in ((rec.get("rag") or {}).get("chunks") or [])}
        return bool(gold & got)
    return None


def build() -> dict:
    records = [json.loads(l) for l in SOURCE.read_text(encoding="utf-8").splitlines() if l.strip()]
    examples = []
    for rec in records:
        state = {"answer": rec["answer"], "route": rec["route"], "trajectory": rec["trajectory"],
                 "standalone_question": rec.get("standalone_question"), "sql": rec.get("sql"), "rag": rec.get("rag"),
                 "verified": rec.get("verified"), "verifier": {"removed": rec.get("removed") or []},
                 "usage": rec["usage"], "latency_s": rec["latency_s"]}
        ex = shape_answer(state)
        ex.update(id=rec["id"], question=rec["question"], expected_route=rec["expected_route"],
                  matches_gold=matches_gold(rec), source="precomputed", source_note=RUN_NOTE)
        examples.append(ex)
    return {"description": "Pre-computed answers from a real run on the eval-v1 held-out questions. They cost no quota.",
            "source_run": SOURCE.name, "model": config.GEMINI_MODEL, "examples": examples}


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1, ensure_ascii=False), encoding="utf-8")
    ex = data["examples"]
    print(f"wrote {OUT} with {len(ex)} examples; matched gold: {sum(e['matches_gold'] is True for e in ex)}, "
          f"not matched: {sum(e['matches_gold'] is False for e in ex)}, no gold check: {sum(e['matches_gold'] is None for e in ex)}")


if __name__ == "__main__":
    main()
