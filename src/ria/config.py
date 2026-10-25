"""Central config. Change the model name and paths here only."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --- LLM (Google Gemini via langchain-google-genai) ---
GEMINI_MODEL = "gemini-3.5-flash-lite"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
# Thinking level per role (Gemini docs: gemini-3.5-flash-lite accepts minimal, low, medium, high; default minimal).
# Roles not listed use the model default. Router and SQL agent use "low" for speed. The RAG agent uses "medium"
# because the lighter model at its default (minimal) wrongly said "not covered" on a question it should answer.
THINKING_LEVELS = {"router": "low", "sql": "low", "rag": "medium"}
LLM_TEMPERATURE = 0.0
# Local LLM response cache for development. One flag: set RIA_LLM_CACHE=0 (or this to False) to turn it off.
# Default is ON (tests, examples). Final evals and latency runs must turn it OFF.
LLM_CACHE_ENABLED = os.getenv("RIA_LLM_CACHE", "1") != "0"
LLM_MAX_ATTEMPTS = 5  # Tenacity: retry on 429 and 5xx only
LLM_BACKOFF_MULTIPLIER = 4  # seconds; waits are random up to 4, 8, 16, 32 (free tier minute limits need ~40s)
# Free tier allows 15 requests per minute per model. A small gap between calls avoids most 429s. Set 0 on a paid plan.
MIN_SECONDS_BETWEEN_CALLS = 4.5

# --- Paths ---
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
DB_DIR = DATA_DIR / "db"
CHROMA_DIR = DATA_DIR / "chroma"
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm"
CHUNKS_PATH = CHROMA_DIR / "chunks.json"
RESULTS_MD = ROOT / "docs" / "results.md"

# --- Retrieval ---
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHROMA_COLLECTION = "policy_chunks"
TOP_K = 5
CANDIDATES = 20  # candidates taken from each retriever before fusion
RRF_K = 60
RERANK_POOL = 20  # reranker rescoring pool: top 20 candidates, then keep top_k
CHUNK_TOKENS = 400  # fallback chunker only
CHUNK_OVERLAP = 60
