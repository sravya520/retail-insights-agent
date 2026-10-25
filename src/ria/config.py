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
THINKING_LEVELS = {"router": "low", "sql": "low", "rag": "medium", "judge": "medium", "rewrite": "low"}
LLM_TEMPERATURE = 0.0
# LLM judge (Phase 4). Prefer a model different from the answering model. The pool falls back to the next
# model when one runs out of free daily quota. The last entry is the answering model itself (self-judging).
JUDGE_MODELS = ["gemini-3.6-flash", "gemini-3.7-flash", GEMINI_MODEL]
# Local LLM response cache for development. One flag: set RIA_LLM_CACHE=0 (or this to False) to turn it off.
# Default is ON (tests, examples). Final evals and latency runs must turn it OFF.
LLM_CACHE_ENABLED = os.getenv("RIA_LLM_CACHE", "1") != "0"
# Price table, USD per 1 million tokens. EDIT THIS when prices change.
# Source: "Standard" table on https://ai.google.dev/gemini-api/docs/pricing. VERIFIED on 2026-10-25 against the raw
# HTML of that page (all three rows below matched). Prices change, so re-check before relying on them. The free tier bills nothing. These are the paid-tier prices, so cost
# figures show what the same usage would cost on a paid plan. Output price includes thinking tokens.
# gemini-3.6-flash and gemini-3.8-flash are listed at $0.75 / $3.75 / $0.075 until 2026-12-31, then double.
PRICES = {
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50, "cached_input": 0.03},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75, "cached_input": 0.075},
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75, "cached_input": 0.075},
}
# Prompt layout: "stable_first" (production) or "question_first" (baseline used only by the caching experiment).
PROMPT_LAYOUT = "stable_first"
# Query rewrite before policy retrieval (adopted only if the dev experiment says it helps, see docs/results.md).
QUERY_REWRITE = False
LLM_MAX_ATTEMPTS = 5  # Tenacity: retry on 429 and 5xx only
LLM_BACKOFF_MULTIPLIER = 4  # seconds; waits are random up to 4, 8, 16, 32 (free tier minute limits need ~40s)
# Free tier allows 15 requests per minute per model. A small gap between calls avoids most 429s. Set 0 on a paid plan.
MIN_SECONDS_BETWEEN_CALLS = 4.5

# --- Public demo limits (protect the free Gemini quota) ---
# Each live question costs 1 to 6 model calls (about 3 on average). The free tier of the answering model allowed
# about 440 calls on 2026-10-25, so 60 questions a day leaves room for the nightly job and for experiments.
DEMO_SESSION_LIMIT = int(os.getenv("RIA_SESSION_LIMIT", "5"))   # live questions per session per quota day
DEMO_DAILY_LIMIT = int(os.getenv("RIA_DAILY_LIMIT", "60"))      # live questions for the whole demo per quota day
MAX_QUESTION_CHARS = 300

# --- Paths ---
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
DB_DIR = DATA_DIR / "db"
CHROMA_DIR = DATA_DIR / "chroma"
LLM_CACHE_DIR = DATA_DIR / "cache" / "llm"
DEPLOY_DIR = DATA_DIR / "deploy"            # parquet copies of the sales tables, committed so a host can rebuild the database
EXAMPLES_PATH = DATA_DIR / "examples" / "examples_v1.json"
USAGE_STATE_PATH = DATA_DIR / "state" / "usage.json"
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
