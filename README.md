# Retail Insights Agent

A business user asks questions in plain English about a retail company. A LangGraph multi-agent system answers them. Data questions go to a SQL agent. Policy questions go to a RAG agent that cites its sources. A verifier checks every answer before it is shown.

Design principle: the LLM decides and explains. Code does the maths and checks the facts.

> **Status: Phase 2 of 6 done (documents, RAG, sales database, SQL agent).** The LangGraph workflow, the full eval harness, token economics, the API and the UI come in later phases.

## Important: the policy documents are synthetic

Northwind Gifts is a fictional company. The six policy documents in `data/docs/` were written for this demo. They are not real company policies. The rules are made up so that they are specific and testable.

The documents are:

1. Returns and Refunds
2. Warranty and Product Quality
3. Shipping and Delivery
4. Wholesale and B2B Terms
5. Loyalty Programme
6. Customer Data and Privacy

Each one is written in Markdown (`data/docs/*.md`) and rendered to PDF with `reportlab`. Ingestion reads the PDFs, so page numbers in citations are real PDF pages.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows. On Linux or macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install -e . --no-build-isolation
cp .env.example .env          # then add your GOOGLE_API_KEY
```

Behind an antivirus that inspects HTTPS (for example AVG), `pip` and the Hugging Face download can fail with `CERTIFICATE_VERIFY_FAILED`. Point `PIP_CERT`, `REQUESTS_CA_BUNDLE` and `SSL_CERT_FILE` at a CA bundle that includes your antivirus certificate.

## Phase 1: run it

```bash
python -m ria.make_policy_pdfs      # Markdown -> PDF
python -m ria.ingest_docs           # PDF -> chunks -> embeddings -> Chroma
python evals/retrieval_eval.py      # recall@5 and MRR for four retrieval methods
python evals/rag_examples.py        # example answers with citations (needs GOOGLE_API_KEY)
pytest
```

Results and the commands that produced them are in [docs/results.md](docs/results.md).

## How the RAG part works

- Chunking is by numbered clause (for example 4.2). Each chunk keeps `doc`, `section`, `clause_id` and `page`.
- Embeddings: `BAAI/bge-small-en-v1.5`, stored in a persistent Chroma collection.
- Hybrid search: BM25 plus vector search, fused with Reciprocal Rank Fusion (k=60). An optional cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) can be switched on.
- The RAG agent answers only from the top 5 chunks. Every factual sentence ends with a citation like `[Returns and Refunds §4.2, p.2]`. If the chunks do not hold the answer, it says "The policy documents do not cover this."

## Phase 2: sales database and SQL agent

```bash
python -I -m ria.sql_db             # load the UCI Online Retail II dataset into data/db/sales.duckdb
python evals/build_sql_gold.py      # run the gold SQL and store the gold answers
python evals/sql_eval.py            # execution accuracy of the SQL agent on 15 questions
python evals/call_latency.py        # per-call latency and tokens from logs/llm_calls.jsonl
```

The dataset is downloaded by hand from https://archive.ics.uci.edu/dataset/502/online+retail+ii into `data/raw/online_retail_ii/`. It is git-ignored.

The LLM writes the SQL. Code decides whether it may run: `sqlglot` allows one SELECT only, over known tables and columns only, with LIMIT 200, a 5 second timeout and a read-only connection. If the query fails, the error goes back to the LLM once for a repair.

## Model and free tier limits

The model name is set in one place, `src/ria/config.py` (`GEMINI_MODEL`). The free Gemini tier has per-minute and per-day request limits that differ by model. The code waits between calls to stay under 15 requests per minute. If you hit a daily limit, the call fails with a clear message. Thinking level per role is set in `THINKING_LEVELS`. See `docs/results.md` for what was measured.

## Dev and held-out questions

Questions used to build or tune the system are marked `dev`. See `evals/DEV_SPLIT.md`. The final eval will report held-out questions separately.
