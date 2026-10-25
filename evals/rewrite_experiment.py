"""Does a query rewrite step improve policy retrieval? Dev questions only (the 20 policy questions p01 to p20).

DECISION RULE, fixed before the first run: adopt the fused search (original question + rewritten query) only if
dev MRR@5 improves by at least 0.03 over plain hybrid search AND recall@5 does not drop. Otherwise leave
config.QUERY_REWRITE off. The cost is one extra LLM call per policy question.

Variants:
  baseline    hybrid search on the original question (BM25 + vector, RRF)
  rewrite     hybrid search on the rewritten query only
  fused       BM25 and vector rankings of both queries fused with RRF
The LLM response cache is ON here (this is a development experiment on dev questions, not a final eval).

Run: python evals/rewrite_experiment.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria.llm import get_llm  # noqa: E402
from ria.rag_agent import rewrite_query  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.retrieval import Retriever  # noqa: E402
from ria.tracing import LLM_CALLS_PATH  # noqa: E402

MIN_MRR_GAIN = 0.03


def first_rank(hits, gold) -> int | None:
    gold_set = {(g["doc"], g["clause_id"]) for g in gold}
    for rank, h in enumerate(hits, start=1):
        if (h.chunk.doc, h.chunk.clause_id) in gold_set:
            return rank
    return None


def metrics(ranks: list[int | None]) -> tuple[float, float]:
    return sum(r is not None for r in ranks) / len(ranks), sum(1 / r for r in ranks if r) / len(ranks)


def main() -> None:
    questions = [json.loads(l) for l in (ROOT / "evals" / "policy_questions.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    retriever = Retriever()
    llm = get_llm("rewrite")
    log_start = len(LLM_CALLS_PATH.read_text(encoding="utf-8-sig").splitlines()) if LLM_CALLS_PATH.exists() else 0
    ranks = {"baseline": [], "rewrite": [], "fused": []}
    rewrites = {}
    for q in questions:
        rq = rewrite_query(q["question"], llm)
        rewrites[q["id"]] = rq
        ranks["baseline"].append(first_rank(retriever.search(q["question"], mode="hybrid", top_k=5), q["gold"]))
        ranks["rewrite"].append(first_rank(retriever.search(rq, mode="hybrid", top_k=5), q["gold"]))
        ranks["fused"].append(first_rank(retriever.search_multi([q["question"], rq], top_k=5), q["gold"]))
    res = {k: metrics(v) for k, v in ranks.items()}
    base, fused = res["baseline"], res["fused"]
    adopt = fused[1] >= base[1] + MIN_MRR_GAIN and fused[0] >= base[0]
    calls = [json.loads(l) for l in LLM_CALLS_PATH.read_text(encoding="utf-8-sig").splitlines()[log_start:] if l.strip()]
    rw = [c for c in calls if c["node"] == "query_rewrite" and not c.get("cache_hit") and "error" not in c]
    cost_note = (f"Each rewrite call averaged {statistics.mean(c['input_tokens'] for c in rw):.0f} input and "
                 f"{statistics.mean(c['output_tokens'] for c in rw):.0f} output tokens "
                 f"(${statistics.mean(c['cost_usd'] for c in rw):.6f}, {statistics.median(c['latency_s'] for c in rw):.1f} s median) over {len(rw)} real calls."
                 if rw else "All rewrite calls came from the LLM cache, so no new cost figures were measured in this run.")
    rows = "\n".join(f"| {k} | {r:.2f} | {m:.3f} |" for k, (r, m) in res.items())
    changed = "\n".join(f"- {q['id']} ({'baseline ' + str(ranks['baseline'][i]) + ' -> fused ' + str(ranks['fused'][i])}): {q['question']} | rewritten as: {rewrites[q['id']]}"
                        for i, q in enumerate(questions) if ranks["baseline"][i] != ranks["fused"][i])
    update_section("phase5_rewrite_experiment", f"""### Query rewrite experiment (dev policy questions p01 to p20, retrieval only)

Command: `python evals/rewrite_experiment.py`. Model for the rewrite: `{config.GEMINI_MODEL}` (thinking low). LLM cache on (dev experiment).
Decision rule written before the run: adopt the fused search only if MRR@5 rises by at least {MIN_MRR_GAIN} and recall@5 does not drop.

| Variant | Recall@5 | MRR@5 |
|---|---|---|
{rows}

**Verdict: {'ADOPT fused search (rule met).' if adopt else 'DO NOT ADOPT (rule not met).'}** {cost_note}

Questions whose gold rank changed between baseline and fused (rank 1 is best, None means not in the top 5):

{changed or '- none'}

These are dev questions, so this result is optimistic. Whether it holds on unseen questions is measured on eval v2.
""")
    print(res, "adopt =", adopt)
    print(changed)


if __name__ == "__main__":
    main()
