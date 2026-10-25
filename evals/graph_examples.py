"""Phase 3 examples: one run per route, a two-turn memory example, and a verifier fault-injection test.

Real API calls (cache on by default, so a rerun costs nothing): data + follow-up, policy, both, out_of_scope.
The verifier test at the end uses NO model call. It adds a fabricated number and a fake citation to a real
answer, to show that the verifier removes them. It is labelled as an injected fault, not as model behaviour.

Run: python evals/graph_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria.graph import ask, build_graph, default_services  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.tracing import LLM_CALLS_PATH  # noqa: E402
from ria.verifier import verify_answer  # noqa: E402

RUNS = [
    ("data (turn 1 of the memory example)", "demo-memory", "What were the top 5 products by revenue in November 2011?"),
    ("data follow-up (turn 2, same thread_id)", "demo-memory", "And in December?"),
    ("policy (dev question p05)", "demo-policy", "Can a wholesale customer return a damaged item after 40 days?"),
    ("both", "demo-both", "How many orders were returned in December 2010, and how long do retail customers have to return an item if they change their mind?"),
    ("out_of_scope", "demo-oos", "What is the weather in London tomorrow?"),
]


def call_counts() -> tuple[int, int]:
    """(model calls, cache hits) logged so far."""
    if not LLM_CALLS_PATH.exists():
        return 0, 0
    lines = [json.loads(l) for l in LLM_CALLS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    return sum(not l.get("cache_hit") for l in lines), sum(bool(l.get("cache_hit")) for l in lines)


def main() -> None:
    app = build_graph(default_services())
    blocks, outputs = [], {}
    for label, thread, question in RUNS:
        before = call_counts()
        out = ask(app, question, thread)
        after = call_counts()
        outputs[label] = out
        removed = (out.get("verifier") or {}).get("removed", [])
        sql = (out.get("sql") or {}).get("sql")
        print(f"[{label}] route={out['route']} trajectory={out['trajectory']} verified={out['verified']} removed={len(removed)}", flush=True)
        blocks.append(
            f"**{label}**\n\n"
            f"- Thread: `{thread}`\n- Question: {question}\n"
            f"- Standalone question from the router: {out['standalone_question']}\n"
            f"- Route: `{out['route']}`\n- Trajectory: `{out['trajectory']}`\n"
            f"- Verifier: verified = {out['verified']}, sentences removed = {len(removed)}"
            + (f" ({json.dumps(removed)})" if removed else "") + "\n"
            + (f"- SQL: `{sql}`\n" if sql else "")
            + f"- Model calls: {after[0] - before[0]} real, {after[1] - before[1]} from cache. Wall time {out['latency_s']} s.\n\n"
            f"Answer:\n\n> " + out["answer"].replace("\n", "\n> "))

    # Verifier fault injection on the real "both" answer (no model call).
    both = outputs["both"]
    tampered = (both["draft"] + " Revenue also grew 87% on last year. "
                "Wholesale buyers can return goods for 90 days [Returns and Refunds §9.9, p.3].")
    res = verify_answer(tampered, question=both["standalone_question"], sql_rows=both["sql"]["rows"],
                        chunks=both["rag"]["chunks"])
    print("injected fault removed:", [(r.reason, r.detail) for r in res.removed])
    injected = (
        "**Verifier test with an injected fault (not model output)**\n\n"
        "I took the real draft answer of the `both` run and appended two made-up sentences: one with a number that is in no "
        "data or policy text (87%), and one with a citation that matches no retrieved chunk (§9.9).\n\n"
        f"- Removed: {json.dumps([vars(r) for r in res.removed])}\n- verified = {res.verified}\n\n"
        "Answer after verification:\n\n> " + res.answer.replace("\n", "\n> ")
        + f"\n\nThe cleaned answer equals the original draft: {res.answer.strip() == both['draft'].strip()}.")

    update_section("phase3_examples", f"""### Graph example runs (LangGraph, memory, verifier)

Command: `python evals/graph_examples.py`. Model: `{config.GEMINI_MODEL}` for the router, SQL agent, RAG agent and combiner
(thinking: router low, SQL low, RAG and combiner medium). LLM cache: {'on' if config.LLM_CACHE_ENABLED else 'off'}.
Calls marked "from cache" were answered by the local development cache from an earlier identical call, so the wall time of those runs is not a latency measurement.

""" + "\n\n---\n\n".join(blocks) + "\n\n---\n\n" + injected)


if __name__ == "__main__":
    main()
