# Commit hash map

On 2026-10-25 every commit was rewritten to set the author and committer to the repository owner (sravya520). Only the author and committer changed. File contents and commit order are identical. Docs written before the rewrite cite the old hashes (for example the held-out questions commit `fc2973a` and the v2 questions commit `da8dea5`). This table maps each old hash to the new one.

| Old hash | New hash | Commit |
|---|---|---|
| `ce5c77a` | `d0bd549` | Phase 1: synthetic policy docs, PDF ingestion, hybrid retrieval, RAG agent, retrieval eval |
| `c676389` | `ad20b7c` | Phase 1: real example answers, model name update, prompt region rule |
| `c36cb53` | `e8d809e` | Phase 2: DuckDB sales database, guarded SQL agent, SQL eval; model/latency/quota fixes |
| `40f17a4` | `3aef927` | Phase 3: LangGraph workflow with memory, deterministic verifier, LLM dev cache |
| `2c343c1` | `d9c74b0` | results: note Phase 2 SQL eval predates Phase 3 guardrail |
| `fc2973a` | `c1916af` | Phase 4: commit held-out questions (20) and dev set BEFORE any eval run; narrow SQL constant guardrail to numeric literals |
| `f3b43ee` | `b4dd122` | Phase 4: evaluation harness, gates, LLM judge, human label template, results |
| `6ac77f2` | `95b1f38` | Phase 5: tracing with cost, request totals, SQL repair log, optional Langfuse, token report, caching experiment, nightly workflow; freeze eval-v1 section |
| `da8dea5` | `73ad350` | Eval v2: commit fresh held-out set (20 questions) BEFORE any dev fixes and before any run |
| `83cea5a` | `af557d3` | Dev fixes (router prompt, RAG prompt), query rewrite built but not adopted, exit code 2 for incomplete runs; eval v2 status |
| `27469bb` | `dd414bd` | Phase 6: FastAPI, Streamlit UI with full answer details, usage limits, pre-computed examples, deploy files, README, DEPLOY guide |
| `2448936` | `dc33114` | Pin LF line endings for shell scripts and the Dockerfile |
| `60403bf` | `d7b1cc7` | Phase 6: compact metrics table and wrapped SQL in the UI, Phase 6 results and measured memory, deploy guide updates |
| `b7e60db` | `a3a3ea9` | Add docs/resume_facts.md: traced resume facts and audit of untraceable claims |
| `0cd31c5` | `aef0c39` | Docs: fix untraceable claims in README and results.md |
| `388a006` | `f3c2bb8` | Fix: raw API error text can no longer become an answer |
| `87e6bb0` | `1361a80` | Update docs/resume_facts.md to match the fixed docs and the bug fix |
| `48a634c` | `8be3044` | Docs: nightly dry run used 27 model calls (from its saved meta.json), not 30 |
