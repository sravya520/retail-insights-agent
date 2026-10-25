"""RAG agent: answers only from retrieved policy chunks, with a citation on every factual sentence."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ria import config
from ria.llm import get_llm, invoke_with_retry
from ria.retrieval import Hit, Retriever

NOT_COVERED = "The policy documents do not cover this."

CITATION_RE = re.compile(r"\[([^\[\]§]+?)\s*§\s*([\w.]+),\s*p\.\s*(\d+)\]")

SYSTEM_PROMPT = f"""You answer questions about the policies of Northwind Gifts, a UK online gift retailer.
Use ONLY the numbered context passages below. Do not use outside knowledge.

Rules:
1. Every factual sentence must end with one or more citations copied exactly from the passage headers, for example [Returns and Refunds §4.2, p.2].
2. Cite only passages that support the sentence. Never invent a citation.
3. If a question depends on whether the customer is retail or wholesale and this is not stated, give the rule for each.
4. You may use common knowledge only to map a place to a region named in the passages (for example, Spain is in the EU, France is in the EU, Japan is rest of world). All rules, numbers and conditions must come from the passages.
5. If the passages do not contain the answer, reply with exactly: {NOT_COVERED}
6. Be short and clear. Use plain language. Do not add a sources list at the end."""


@dataclass(frozen=True)
class Citation:
    doc: str
    clause_id: str
    page: int


@dataclass
class RagAnswer:
    question: str
    answer: str
    hits: list[Hit]
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[Citation] = field(default_factory=list)


def parse_citations(text: str) -> list[Citation]:
    """Extract citations like [Returns and Refunds §4.2, p.2], in order, without duplicates."""
    seen: list[Citation] = []
    for doc, clause_id, page in CITATION_RE.findall(text):
        cite = Citation(doc.strip(), clause_id.strip().rstrip("."), int(page))
        if cite not in seen:
            seen.append(cite)
    return seen


def split_valid(citations: list[Citation], hits: list[Hit]) -> tuple[list[Citation], list[Citation]]:
    """Split citations into those that match a retrieved chunk (doc + clause_id) and those that do not."""
    known = {(h.chunk.doc, h.chunk.clause_id) for h in hits}
    valid = [c for c in citations if (c.doc, c.clause_id) in known]
    invalid = [c for c in citations if (c.doc, c.clause_id) not in known]
    return valid, invalid


def format_context(hits: list[Hit]) -> str:
    return "\n\n".join(f"Passage {i}: [{h.label}]\n{h.chunk.text}" for i, h in enumerate(hits, start=1))


def answer_policy_question(question: str, retriever: Retriever | None = None, llm=None,
                           rerank: bool = False, top_k: int = config.TOP_K) -> RagAnswer:
    retriever = retriever or Retriever()
    hits = retriever.search(question, mode="hybrid", top_k=top_k, rerank=rerank)
    llm = llm or get_llm()
    messages = [("system", SYSTEM_PROMPT),
                ("human", f"Context passages:\n\n{format_context(hits)}\n\nQuestion: {question}")]
    reply = invoke_with_retry(llm, messages)
    content = reply.content
    text = content if isinstance(content, str) else "".join(
        p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    text = text.strip()
    valid, invalid = split_valid(parse_citations(text), hits)
    return RagAnswer(question, text, hits, valid, invalid)
