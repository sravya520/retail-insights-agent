"""Deterministic verifier. No LLM. Code checks the facts.

Rules, applied sentence by sentence:
  1. Every citation like [Doc §4.2, p.2] must match a retrieved chunk (same doc and clause id).
     A sentence with a citation that matches nothing is removed.
  2. Every number must be supported. A number is supported when it
       * appears in the SQL result rows (with rounding tolerance), or
       * appears in the user's question (the answer may repeat it), or
       * appears in the text of a retrieved chunk that the same sentence cites, or
       * equals the number of rows returned.
     A sentence with an unsupported number is removed.
  3. Removed sentences are returned in `removed` with the reason, so they can be logged.

Not checked (known limits): numbers written as words ("five"), dates written as words,
and whether a cited chunk really says what the sentence says (meaning is not checked, only numbers and ids).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Iterable, Sequence

from ria.rag_agent import BRACKET_RE, Citation, parse_citations

NUM_RE = re.compile(
    r"(?<![\w/.])[£$€]?(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(\s?(?:%|(?:percent|k|m|million|thousand|bn|billion)\b))?(?!\w)", re.I)
SCALES = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "bn": 1e9, "billion": 1e9}
LIST_MARKER_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z£$€(0-9])")

FALLBACK_ANSWER = "I could not produce a verified answer to this question."


@dataclass(frozen=True)
class Number:
    value: float          # value after scaling (1.2m -> 1200000)
    tolerance: float      # half of the last written digit, after scaling
    is_percent: bool = False


def extract_numbers(text: str) -> list[Number]:
    """Find numbers in text. Commas, currency signs, %, k, m and million are understood."""
    found = []
    for m in NUM_RE.finditer(text):
        raw, suffix = m.group(1), (m.group(2) or "").strip().lower()
        decimals = len(raw.split(".")[1]) if "." in raw else 0
        scale = SCALES.get(suffix, 1.0)
        value = float(raw.replace(",", "")) * scale
        found.append(Number(value, 0.5 * scale * 10 ** -decimals + 1e-9, suffix in ("%", "percent")))
    return found


def _numbers_in_cell(cell: Any) -> list[float]:
    if cell is None or isinstance(cell, bool):
        return []
    if isinstance(cell, (int, float, Decimal)):
        return [float(cell)]
    return [n.value for n in extract_numbers(str(cell))]


def support_values(rows: Sequence[Sequence[Any]]) -> list[float]:
    """All numbers a SQL result can support: numeric cells, numbers inside text cells, and the row count."""
    values = [float(len(rows))] if rows else []
    for row in rows:
        for cell in row:
            values.extend(_numbers_in_cell(cell))
    return values


def is_supported(number: Number, supports: Iterable[float]) -> bool:
    for s in supports:
        if abs(s - number.value) <= number.tolerance + 1e-9 * abs(s):
            return True
        if number.is_percent and abs(s * 100 - number.value) <= number.tolerance:  # ratio 0.154 written as 15.4%
            return True
    return False


def split_sentences(text: str) -> list[tuple[str, list[str]]]:
    """Split into lines, then sentences. Returns (list marker, sentences) per line.

    A citation after the full stop stays with its sentence. A marker like "1." or "-" is kept apart.
    """
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        marker = LIST_MARKER_RE.match(line)
        prefix = marker.group(0).strip() + " " if marker else ""
        out.append((prefix, SENTENCE_SPLIT_RE.split(line[len(marker.group(0)):] if marker else line)))
    return out


@dataclass
class Removed:
    sentence: str
    reason: str   # "unsupported_number" or "invalid_citation"
    detail: str


@dataclass
class VerifierResult:
    answer: str
    removed: list[Removed] = field(default_factory=list)
    verified: bool = True            # True when nothing had to be removed
    sentences_checked: int = 0
    fallback_used: bool = False      # True when every sentence was removed
    uncited_sentences: int = 0       # policy sentences without a citation (reported, not removed)

    def to_log(self) -> dict[str, Any]:
        return {"verified": self.verified, "sentences_checked": self.sentences_checked,
                "removed": [vars(r) for r in self.removed], "fallback_used": self.fallback_used,
                "uncited_sentences": self.uncited_sentences}


def _chunk_key(c: dict[str, Any]) -> tuple[str, str]:
    return c["doc"], c["clause_id"]


def verify_answer(answer: str, question: str = "", sql_rows: Sequence[Sequence[Any]] | None = None,
                  chunks: Sequence[dict[str, Any]] | None = None) -> VerifierResult:
    """Check `answer`. `chunks` are dicts with doc, clause_id and text (the retrieved policy chunks)."""
    sql_rows = sql_rows or []
    chunks = chunks or []
    chunk_by_key = {_chunk_key(c): c for c in chunks}
    sql_supports = support_values(sql_rows)
    question_numbers = extract_numbers(question)
    result = VerifierResult(answer="")
    kept_lines: list[str] = []

    for prefix, line_sentences in split_sentences(answer):
        kept = []
        for sentence in line_sentences:
            result.sentences_checked += 1
            citations: list[Citation] = parse_citations(sentence)
            bad = [c for c in citations if (c.doc, c.clause_id) not in chunk_by_key]
            if bad:
                result.removed.append(Removed(sentence, "invalid_citation",
                                              ", ".join(f"{c.doc} §{c.clause_id}" for c in bad)))
                continue
            # Numbers are checked on the text without citation brackets (clause ids and pages are not claims).
            body = BRACKET_RE.sub(" ", sentence)
            cited_numbers = [v for c in citations for v in
                             (n.value for n in extract_numbers(chunk_by_key[(c.doc, c.clause_id)]["text"]))]
            unsupported = [n for n in extract_numbers(body)
                           if not (is_supported(n, sql_supports) or is_supported(n, cited_numbers)
                                   or is_supported(n, [q.value for q in question_numbers]))]
            if unsupported:
                result.removed.append(Removed(sentence, "unsupported_number",
                                              ", ".join(f"{n.value:g}" for n in unsupported)))
                continue
            if chunks and not citations and not extract_numbers(body) and not sql_rows:
                result.uncited_sentences += 1
            kept.append(sentence)
        if kept:
            kept_lines.append(prefix + " ".join(kept))

    result.verified = not result.removed
    if kept_lines:
        result.answer = "\n".join(kept_lines)
    else:
        result.answer, result.fallback_used = FALLBACK_ANSWER, True
    return result
