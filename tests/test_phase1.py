from pathlib import Path

import pytest
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from ria.ingest_docs import chunk_by_clause, chunk_pdf, window_chunks
from ria.llm import is_retryable
from ria.rag_agent import Citation, parse_citations, split_valid
from ria.retrieval import Hit, rrf_fuse
from ria.ingest_docs import Chunk

PAGE1 = """Test Policy
Intro paragraph.
1. Scope
1.1 First rule. This is the first rule and
it wraps onto a second line.
1.2 Second rule. Cost is £3.95 for most orders.
Northwind Gifts (synthetic document) | Test Policy | Page 1"""

PAGE2 = """2. Returns
2.1 Third rule. This rule starts on page two.
3.95 is not a clause number.
Northwind Gifts (synthetic document) | Test Policy | Page 2"""


def test_chunking_keeps_page_numbers_and_sections():
    chunks = chunk_by_clause([PAGE1, PAGE2], "Test Policy", "test")
    assert [c.clause_id for c in chunks] == ["1.1", "1.2", "2.1"]
    assert [c.page for c in chunks] == [1, 1, 2]
    assert chunks[0].section == "1. Scope"
    assert chunks[2].section == "2. Returns"
    assert "wraps onto a second line" in chunks[0].text
    assert "Page 1" not in chunks[1].text  # footer removed
    assert "3.95 is not a clause number" in chunks[2].text  # not mistaken for a clause


def test_chunking_from_real_pdf_keeps_pages(tmp_path: Path):
    pdf = tmp_path / "mini.pdf"
    c = canvas.Canvas(str(pdf), pagesize=A4)
    for page_lines in ([ "Mini Policy", "1. Scope", "1.1 Alpha. Alpha text."], ["2. Rules", "2.1 Beta. Beta text."]):
        y = 800
        for line in page_lines:
            c.drawString(50, y, line)
            y -= 20
        c.showPage()
    c.save()
    chunks = chunk_pdf(pdf)
    assert [(ch.doc, ch.clause_id, ch.page) for ch in chunks] == [("Mini Policy", "1.1", 1), ("Mini Policy", "2.1", 2)]


def test_real_policy_pdfs_have_titles_and_pages():
    from ria import config
    pdf = config.DOCS_DIR / "returns_and_refunds.pdf"
    if not pdf.exists():
        pytest.skip("run python -m ria.make_policy_pdfs first")
    chunks = {c.clause_id: c for c in chunk_pdf(pdf)}
    assert chunks["4.2"].doc == "Returns and Refunds"
    assert chunks["4.2"].section == "4. Damaged and faulty items"
    assert chunks["1.1"].page == 1 and chunks["9.2"].page == max(c.page for c in chunks.values()) >= 2


def test_window_chunks_overlap_and_pages():
    pages = [" ".join(f"w{i}" for i in range(500)), " ".join(f"x{i}" for i in range(100))]
    chunks = window_chunks(pages, "Doc", "d", size=400, overlap=60)
    assert chunks[0].text.split()[0] == "w0" and chunks[0].page == 1
    assert chunks[1].text.split()[0] == "w340"  # 400 - 60 = step of 340
    assert len(chunks[0].text.split()) == 400


def test_rrf_fusion_order():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [i for i, _ in fused]
    assert ids[:2] == ["a", "b"]  # equal scores, "a" seen first
    assert ids[2:] == ["c", "d"]
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)


def test_rrf_rewards_agreement():
    # "x" is rank 2 in both lists and beats "y" and "z", which are rank 1 in only one list.
    fused = rrf_fuse([["y", "x"], ["z", "x"]], k=60)
    assert fused[0][0] == "x"


def test_parse_citations():
    text = ("Wholesale returns are allowed for 14 days [Returns and Refunds §7.1, p.3]. "
            "A fee applies [Returns and Refunds §7.2, p.3][Wholesale and B2B Terms §6.1, p.2]. "
            "Repeat [Returns and Refunds §7.1, p.3].")
    cites = parse_citations(text)
    assert cites == [Citation("Returns and Refunds", "7.1", 3), Citation("Returns and Refunds", "7.2", 3),
                     Citation("Wholesale and B2B Terms", "6.1", 2)]


def test_parse_citations_ignores_malformed():
    assert parse_citations("See [Returns and Refunds 7.1] and [§7.1, p.3] and [1].") == []


def test_split_valid_flags_fake_citation():
    chunk = Chunk("r:7.1", "Returns and Refunds", "7. Wholesale returns", "7.1", "Wholesale return window", 3, "text")
    hits = [Hit(chunk, 1.0, "hybrid")]
    valid, invalid = split_valid([Citation("Returns and Refunds", "7.1", 3), Citation("Returns and Refunds", "9.9", 1)], hits)
    assert valid == [Citation("Returns and Refunds", "7.1", 3)]
    assert invalid == [Citation("Returns and Refunds", "9.9", 1)]


class _Err(Exception):
    def __init__(self, msg: str, code: int | None = None):
        super().__init__(msg)
        self.code = code


@pytest.mark.parametrize("exc,expected", [
    (_Err("rate limited", 429), True),
    (_Err("boom", 503), True),
    (_Err("bad request", 400), False),
    (_Err("401 Unauthorized, API key invalid"), False),
    (_Err("429 Resource has been exhausted"), True),
    (ValueError("something else"), False),
])
def test_is_retryable(exc, expected):
    assert is_retryable(exc) is expected


class _FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def search(self, query, mode="hybrid", top_k=5, rerank=False):
        return self.hits


class _FakeLLM:
    def __init__(self, text):
        self.text = text
        self.messages = None

    def invoke(self, messages):
        self.messages = messages

        class R:
            content = self.text
        return R()


def _hit():
    chunk = Chunk("r:7.1", "Returns and Refunds", "7. Wholesale returns", "7.1", "Wholesale return window", 3,
                  "7.1 Wholesale return window. 14 days.")
    return Hit(chunk, 1.0, "hybrid")


def test_rag_agent_flags_invented_citation_with_mock_llm():
    from ria.rag_agent import answer_policy_question
    llm = _FakeLLM("Returns take 14 days [Returns and Refunds §7.1, p.3]. Also 99 days [Returns and Refunds §9.9, p.1].")
    res = answer_policy_question("q?", retriever=_FakeRetriever([_hit()]), llm=llm)
    assert res.citations == [Citation("Returns and Refunds", "7.1", 3)]
    assert res.invalid_citations == [Citation("Returns and Refunds", "9.9", 1)]
    assert "[Returns and Refunds §7.1, p.3]" in llm.messages[1][1]  # context header matches the citation format


def test_rag_agent_not_covered_has_no_citations():
    from ria.rag_agent import NOT_COVERED, answer_policy_question
    res = answer_policy_question("q?", retriever=_FakeRetriever([_hit()]), llm=_FakeLLM(NOT_COVERED))
    assert res.answer == NOT_COVERED and res.citations == []


def test_daily_quota_429_is_not_retried():
    msg = "429 RESOURCE_EXHAUSTED. quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier"
    assert is_retryable(_Err(msg, 429)) is False
    assert is_retryable(_Err("429 RESOURCE_EXHAUSTED per minute limit", 429)) is True


def test_parse_citations_merged_bracket():
    text = "Not returnable [Returns and Refunds §2.3, p.1, Returns and Refunds §3.1, p.1]. Also [Shipping and Delivery §3.1, p.1; Shipping and Delivery §3.2, p.1]."
    assert parse_citations(text) == [Citation("Returns and Refunds", "2.3", 1), Citation("Returns and Refunds", "3.1", 1),
                                     Citation("Shipping and Delivery", "3.1", 1), Citation("Shipping and Delivery", "3.2", 1)]
