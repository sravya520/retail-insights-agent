"""Make sure the data a running app needs exists: the DuckDB file and the vector index.

Both can be rebuilt from files committed to the repository, so a fresh checkout on a free host starts without
manual steps. The vector index is built from the policy PDFs (the first run downloads the embedding model).
"""
from __future__ import annotations

from ria import config
from ria.sql_db import DB_PATH, ensure_db


def index_ready() -> bool:
    if not config.CHUNKS_PATH.exists() or not (config.CHROMA_DIR / "chroma.sqlite3").exists():
        return False
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
        return client.get_collection(config.CHROMA_COLLECTION).count() > 0
    except Exception:  # noqa: BLE001 - any problem means "rebuild"
        return False


def ensure_index() -> None:
    if index_ready():
        return
    from ria.ingest_docs import ingest
    ingest()


def ensure_data() -> dict[str, str]:
    """Build whatever is missing. Returns what was done, for logging."""
    done = {}
    if not DB_PATH.exists():
        ensure_db()
        done["database"] = "rebuilt from parquet"
    if not index_ready():
        ensure_index()
        done["vector_index"] = "rebuilt from policy PDFs"
    return done
