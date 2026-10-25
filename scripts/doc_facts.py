"""Print the derived numbers that docs/results.md, README.md and docs/resume_facts.md cite. No API calls.

Reads only saved files. The last block needs logs/ (git-ignored), so it works only on the machine that ran the experiment.

Run: python scripts/doc_facts.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria.sql_compare import results_match  # noqa: E402

RUNS = ROOT / "evals" / "runs"
PRICE = config.PRICES["gemini-3.5-flash-lite"]


def read(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8-sig").splitlines() if l.strip()]


def heldout_sql_accuracy() -> None:
    print("1. Held-out SQL execution accuracy on the same questions, by run")
    for name in ("run-phase4/records.jsonl", "caching-question_first.jsonl", "caching-stable_first.jsonl"):
        recs = [r for r in read(RUNS / name) if r.get("split") == "heldout" and r.get("gold_answer") is not None]
        ok = [r["id"] for r in recs if r["sql"] and not r["sql"]["error"]
              and results_match(r["gold_answer"], r["sql"]["rows"], ordered=r.get("ordered", False))]
        missed = sorted({r["id"] for r in recs} - set(ok))
        print(f"   {name}: {len(ok)} of {len(recs)}" + (f" (missed: {', '.join(missed)})" if missed else ""))
        for r in recs:
            if r["id"] in missed:
                print(f"      {r['id']}: agent rows {r['sql']['rows'][:2]}, gold {r['gold_answer'][:2]}, agent SQL: {r['sql']['sql']}")


def eval_v1_details() -> None:
    print("5. Eval v1 details behind the narrative (run run-phase4)")
    recs = read(RUNS / "run-phase4" / "records.jsonl")
    for r in recs:
        if r.get("trajectory") != r["expected_trajectory"]:
            print(f"   wrong path: {r['id']} ({r['split']}) expected {r['expected_route']}, got {r['route']}: {r['question']}")
            print(f"      answer starts: {r['answer'][:110]}")
    for r in recs:
        j = r.get("judge")
        if j and j["faithfulness"] < 5:
            print(f"   judge faithfulness {j['faithfulness']}: {r['id']} ({r['split']}, judge {j['model']}): {j['unsupported_claims'][:120]}")
    repaired = [r["id"] for r in recs if (r.get("sql") or {}).get("repaired")]
    print(f"   SQL queries that needed the repair attempt: {repaired or 'none'}")


def cost_shares() -> None:
    print("2. Input and output cost per caching-experiment layout (prices from config.PRICES)")
    for layout in ("question_first", "stable_first"):
        recs = read(RUNS / f"caching-{layout}.jsonl")
        inp = sum(r["usage"]["input_tokens"] for r in recs)
        out = sum(r["usage"]["output_tokens"] for r in recs)
        cost_in, cost_out = inp * PRICE["input"] / 1e6, out * PRICE["output"] / 1e6
        total = cost_in + cost_out
        print(f"   {layout}: {len(recs)} requests, input {inp} tokens, output {out} tokens, "
              f"output share of cost {100 * cost_out / total:.1f}%, "
              f"most saved if every input token were cached {100 * 0.9 * cost_in / total:.1f}%")


def quota_limit() -> None:
    print("3. Free-tier daily limit stated in the API's own error message (saved in a run record)")
    found, quota_ids = set(), set()
    for r in read(RUNS / "run-dev-after-fixes" / "records.jsonl"):
        text = json.dumps(r)
        found |= set(re.findall(r"limit: (\d+), model: ([\w.-]+)", text))
        quota_ids |= set(re.findall(r"(GenerateRequestsPer\w+)", text))
    for limit, model in sorted(found):
        print(f"   {model}: limit {limit} (quota ids in the same message: {', '.join(sorted(quota_ids))})")


def thinking_share() -> None:
    print("4. Thinking (reasoning) tokens in the stable_first run (local only: needs logs/)")
    calls_path, traces_path = ROOT / "logs" / "llm_calls.jsonl", ROOT / "logs" / "traces.jsonl"
    if not calls_path.exists() or not traces_path.exists():
        print("   skipped: logs/ not present on this machine")
        return
    end = [t["ts"] for t in read(traces_path) if t.get("label") == "cache-stable_first"][-1]
    nodes = {"router", "sql_agent.write", "sql_agent.repair", "sql_agent.explain", "rag_agent", "combiner"}
    window = [c for c in read(calls_path) if c["node"] in nodes and c["ts"] <= end][-54:]
    inp = sum(c["input_tokens"] for c in window)
    out = sum(c["output_tokens"] for c in window)
    reasoning = sum(c["reasoning_tokens"] for c in window)
    total = inp * PRICE["input"] + out * PRICE["output"]
    print(f"   {len(window)} calls, output {out} tokens, of which reasoning {reasoning} "
          f"({100 * reasoning / out:.1f}% of output, {100 * reasoning * PRICE['output'] / total:.1f}% of cost)")


if __name__ == "__main__":
    heldout_sql_accuracy()
    cost_shares()
    quota_limit()
    thinking_share()
    eval_v1_details()
