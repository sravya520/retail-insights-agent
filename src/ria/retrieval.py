"""Hybrid retrieval: BM25 + vector search fused with Reciprocal Rank Fusion, optional reranker."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from rank_bm25 import BM25Okapi

from ria import config
from ria.ingest_docs import Chunk

Mode = Literal["bm25", "vector", "hybrid"]


@dataclass
class Hit:
    chunk: Chunk
    score: float
    mode: str

    @property
    def label(self) -> str:
        """Citation label for this chunk, e.g. 'Returns and Refunds §4.2, p.2'."""
        return f"{self.chunk.doc} §{self.chunk.clause_id}, p.{self.chunk.page}"


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def rrf_fuse(rankings: list[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion. score(d) = sum over rankings of 1 / (k + rank), rank starts at 1.

    Ties are broken by the order in which the id was first seen, so the result is deterministic.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    first_seen = {doc_id: i for i, doc_id in enumerate(scores)}
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))


class Retriever:
    def __init__(self) -> None:
        raw = json.loads(config.CHUNKS_PATH.read_text(encoding="utf-8"))
        self.chunks = [Chunk(**c) for c in raw]
        self.by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25Okapi([tokenize(c.embed_text) for c in self.chunks])

    @cached_property
    def _embedder(self):
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(config.EMBED_MODEL)

    @cached_property
    def _collection(self):
        import chromadb
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return client.get_collection(config.CHROMA_COLLECTION)

    @cached_property
    def _reranker(self):
        from sentence_transformers import CrossEncoder
        return CrossEncoder(config.RERANK_MODEL)

    def _bm25_ranking(self, query: str, n: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(self.chunks)), key=lambda i: (-scores[i], i))[:n]
        return [(self.chunks[i].id, float(scores[i])) for i in order]

    def _vector_ranking(self, query: str, n: int) -> list[tuple[str, float]]:
        emb = self._embedder.encode([config.EMBED_QUERY_PREFIX + query], normalize_embeddings=True)
        res = self._collection.query(query_embeddings=emb.tolist(), n_results=n)
        # Chroma returns cosine distance. Similarity = 1 - distance.
        return [(i, 1.0 - d) for i, d in zip(res["ids"][0], res["distances"][0])]

    def search(self, query: str, mode: Mode = "hybrid", top_k: int = config.TOP_K,
               rerank: bool = False, rerank_pool: int = config.RERANK_POOL) -> list[Hit]:
        """Search. With rerank=True the top `rerank_pool` candidates are rescored and the best `top_k` kept."""
        n = max(config.CANDIDATES, rerank_pool) if rerank else config.CANDIDATES
        if mode == "bm25":
            ranked = self._bm25_ranking(query, n)
        elif mode == "vector":
            ranked = self._vector_ranking(query, n)
        else:
            fused = rrf_fuse([[i for i, _ in self._bm25_ranking(query, n)],
                              [i for i, _ in self._vector_ranking(query, n)]])
            ranked = fused[:n]
        if rerank:
            ranked = ranked[:rerank_pool]
            pairs = [(query, self.by_id[i].embed_text) for i, _ in ranked]
            scores = self._reranker.predict(pairs)
            ranked = sorted(zip((i for i, _ in ranked), map(float, scores)), key=lambda kv: -kv[1])
            mode = f"{mode}+rerank"
        return [Hit(self.by_id[i], s, mode) for i, s in ranked[:top_k]]
