"""Central config. Change the model name and paths here only."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

# --- LLM (Google Gemini via langchain-google-genai) ---
GEMINI_MODEL = "gemini-3.8-flash"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
LLM_TEMPERATURE = 0.0
LLM_MAX_ATTEMPTS = 5  # Tenacity: retry on 429 and 5xx only

# --- Paths ---
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
DB_DIR = DATA_DIR / "db"
CHROMA_DIR = DATA_DIR / "chroma"
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
CHUNK_TOKENS = 400  # fallback chunker only
CHUNK_OVERLAP = 60
