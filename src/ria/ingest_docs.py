"""PDF -> chunks -> embeddings -> Chroma, plus a JSON chunk file used for the BM25 index.

Chunking is by numbered clause (for example 4.2). Each chunk keeps its doc title, section
heading, clause id and the page where the clause starts. If a PDF has no numbered clauses,
we fall back to a sliding window of about 400 tokens with 60 overlap (a word counts as a token).

Run: python -m ria.ingest_docs
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from pypdf import PdfReader

from ria import config

FOOTER_RE = re.compile(r"^Northwind Gifts \(synthetic document\) \|")
SECTION_RE = re.compile(r"^(\d{1,2})\.\s+([A-Z].*)$")
CLAUSE_RE = re.compile(r"^(\d{1,2})\.(\d{1,2})\s+([A-Z].*)$")


@dataclass
class Chunk:
    id: str
    doc: str
    section: str
    clause_id: str
    clause_title: str
    page: int
    text: str

    @property
    def embed_text(self) -> str:
        """Text used for embedding and BM25: doc and section give the clause its context."""
        return f"{self.doc} | {self.section} | {self.text}"

    def metadata(self) -> dict[str, str | int]:
        return {"doc": self.doc, "section": self.section, "clause_id": self.clause_id, "page": self.page}


def read_pdf_pages(path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def _clause_title(text: str) -> str:
    return text.split(". ", 1)[0].strip()


def chunk_by_clause(pages: list[str], doc: str, doc_key: str) -> list[Chunk]:
    """Split page texts into one chunk per numbered clause. Page = page where the clause starts."""
    chunks: list[Chunk] = []
    section_no, last_minor = 0, 0
    section_title = ""
    current: dict | None = None

    def flush() -> None:
        nonlocal current
        if current:
            text = " ".join(current["lines"]).strip()
            chunks.append(Chunk(
                id=f"{doc_key}:{current['clause_id']}", doc=doc, section=section_title,
                clause_id=current["clause_id"], clause_title=_clause_title(text),
                page=current["page"], text=text,
            ))
        current = None

    for page_no, page_text in enumerate(pages, start=1):
        for raw in page_text.splitlines():
            line = raw.strip()
            if not line or FOOTER_RE.match(line):
                continue
            clause = CLAUSE_RE.match(line)
            if clause and int(clause[1]) == section_no and int(clause[2]) == last_minor + 1:
                flush()
                last_minor += 1
                current = {"clause_id": f"{clause[1]}.{clause[2]}", "page": page_no,
                           "lines": [f"{clause[1]}.{clause[2]} {clause[3]}"]}
                continue
            section = SECTION_RE.match(line)
            if section and int(section[1]) == section_no + 1:
                flush()
                section_no, last_minor = int(section[1]), 0
                section_title = f"{section[1]}. {section[2]}"
                continue
            if current:
                current["lines"].append(line)
    flush()
    return chunks


def window_chunks(pages: list[str], doc: str, doc_key: str,
                  size: int = config.CHUNK_TOKENS, overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    """Fallback chunker: sliding word window. Page = page of the first word in the window."""
    words: list[tuple[str, int]] = []
    for page_no, text in enumerate(pages, start=1):
        lines = [ln for ln in text.splitlines() if not FOOTER_RE.match(ln.strip())]
        words.extend((w, page_no) for w in " ".join(lines).split())
    chunks: list[Chunk] = []
    step = size - overlap
    for n, start in enumerate(range(0, max(len(words), 1), step), start=1):
        window = words[start:start + size]
        if not window:
            break
        chunks.append(Chunk(
            id=f"{doc_key}:w{n}", doc=doc, section="", clause_id=f"w{n}", clause_title="",
            page=window[0][1], text=" ".join(w for w, _ in window),
        ))
        if start + size >= len(words):
            break
    return chunks


def chunk_pdf(path: Path) -> list[Chunk]:
    pages = read_pdf_pages(path)
    # reportlab draws the footer first, so pypdf may return it before the title: skip footer lines.
    doc = next(ln.strip() for ln in pages[0].splitlines() if ln.strip() and not FOOTER_RE.match(ln.strip()))
    chunks = chunk_by_clause(pages, doc, path.stem)
    return chunks or window_chunks(pages, doc, path.stem)


def ingest(docs_dir: Path = config.DOCS_DIR) -> list[Chunk]:
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunks: list[Chunk] = []
    for pdf in sorted(docs_dir.glob("*.pdf")):
        chunks.extend(chunk_pdf(pdf))
    if not chunks:
        raise SystemExit(f"No PDFs found in {docs_dir}. Run: python -m ria.make_policy_pdfs")

    model = SentenceTransformer(config.EMBED_MODEL)
    embeddings = model.encode([c.embed_text for c in chunks], normalize_embeddings=True, show_progress_bar=False)

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    try:
        client.delete_collection(config.CHROMA_COLLECTION)
    except Exception:  # noqa: BLE001 - collection may not exist yet
        pass
    collection = client.create_collection(config.CHROMA_COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[c.id for c in chunks], embeddings=embeddings.tolist(),
        documents=[c.text for c in chunks], metadatas=[c.metadata() for c in chunks],
    )
    config.CHUNKS_PATH.write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")
    return chunks


def main() -> None:
    chunks = ingest()
    docs = {c.doc for c in chunks}
    print(f"Ingested {len(chunks)} chunks from {len(docs)} documents into {config.CHROMA_DIR}")


if __name__ == "__main__":
    main()
