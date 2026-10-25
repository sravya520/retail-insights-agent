"""Run the RAG agent on a few questions and save the answers to docs/results.md.

Run: python evals/rag_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria.rag_agent import answer_policy_question  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.retrieval import Retriever  # noqa: E402

QUESTIONS = [
    "Can a wholesale customer return a damaged item after 40 days?",
    "I bought a personalised mug on 5 December. Can I return it in January because I changed my mind?",
    "How much does it cost to ship a £150 order to Spain, and who pays customs charges?",
    "Do you offer a price match against other shops?",
]


def main() -> None:
    retriever = Retriever()
    blocks = []
    for q in QUESTIONS:
        res = answer_policy_question(q, retriever=retriever, rerank=True)
        print(f"Q: {q}\nA: {res.answer}\nvalid={len(res.citations)} invalid={len(res.invalid_citations)}\n")
        blocks.append(
            f"**Q:** {q}\n\n**A:** {res.answer}\n\n"
            f"Citations matched to retrieved chunks: {len(res.citations)}. Not matched: {len(res.invalid_citations)}.")
    update_section("phase1_examples", "### Example answers\n\nCommand: `python evals/rag_examples.py`\n\n" + "\n\n---\n\n".join(blocks))


if __name__ == "__main__":
    main()
