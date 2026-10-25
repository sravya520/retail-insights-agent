"""Phase 1 retrieval eval: recall@5 and MRR@5 for BM25, vector, hybrid and hybrid + reranker.

A question counts as a hit if ANY gold (doc, clause_id) is in the top 5. MRR uses the rank of the first gold in the top 5, 0 if absent.

Run: python evals/retrieval_eval.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria.results_md import update_section  # noqa: E402
from ria.retrieval import Retriever  # noqa: E402

CONFIGS = [("BM25 only", "bm25", False), ("Vector only", "vector", False),
           ("Hybrid (RRF k=60)", "hybrid", False), ("Hybrid + reranker", "hybrid", True)]


def first_gold_rank(hits, gold) -> int | None:
    gold_set = {(g["doc"], g["clause_id"]) for g in gold}
    for rank, h in enumerate(hits, start=1):
        if (h.chunk.doc, h.chunk.clause_id) in gold_set:
            return rank
    return None


def main() -> None:
    questions = [json.loads(line) for line in (ROOT / "evals" / "policy_questions.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    retriever = Retriever()
    rows, misses = [], {}
    for name, mode, rerank in CONFIGS:
        ranks = []
        for q in questions:
            hits = retriever.search(q["question"], mode=mode, top_k=5, rerank=rerank)
            rank = first_gold_rank(hits, q["gold"])
            ranks.append(rank)
            if rank is None:
                misses.setdefault(name, []).append(q["id"])
        recall = sum(r is not None for r in ranks) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        rows.append((name, recall, mrr))
        print(f"{name:22s} recall@5={recall:.3f} MRR@5={mrr:.3f} misses={misses.get(name, [])}")

    table = "\n".join(f"| {n} | {r:.2f} | {m:.3f} |" for n, r, m in rows)
    miss_lines = "\n".join(f"- {n}: {', '.join(ids)}" for n, ids in misses.items()) or "- none"
    update_section("phase1_retrieval", f"""### Retrieval comparison ({len(questions)} policy questions)

Command: `python evals/retrieval_eval.py`

| Method | Recall@5 | MRR@5 |
|---|---|---|
{table}

Questions with no gold clause in the top 5:

{miss_lines}""")


if __name__ == "__main__":
    main()
