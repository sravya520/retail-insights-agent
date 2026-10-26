# Retail Insights Agent: complete project brief

Paste this into another Claude conversation so it knows what this project is, how it was built, what was measured, what went wrong, and what is still open. Every number here is backed by `docs/results.md`, a saved run file, or `python scripts/doc_facts.py` (reads saved files only, no API calls). Status as of 2026-10-26.

## 1. What it is

A portfolio project that shows data-centric agentic AI end to end: retrieval, text-to-SQL, agent orchestration, evaluation and observability.

A business user asks plain-English questions about a retail company. A LangGraph multi-agent system answers:

- **Data questions** ("top 5 products by revenue in November 2011") go to a SQL agent. It writes a read-only DuckDB query that code checks before it runs, then explains the rows.
- **Policy questions** ("can a wholesale customer return a damaged item after 40 days?") go to a RAG agent. It answers only from retrieved policy clauses, with citations like `[Returns and Refunds §7.4, p.2]`.
- **Mixed questions** use both agents, and a combiner joins the answers.
- **Out-of-scope questions** get a fixed polite refusal.
- **A verifier written in plain code** checks every answer. It removes any sentence with a number not found in the SQL rows, the user's question or the cited clause, and any citation that matches no retrieved clause.

Design principle: the LLM decides and explains. Code does the maths and checks the facts.

- **Repository:** https://github.com/sravya520/retail-insights-agent (branch `main`, head `799fe61`)
- **Tag:** `eval-v1` (commit `b4dd122`) freezes the Phase 4 results.
- **Commit hashes:** all commits were rewritten to the owner's author name after the first push, so every hash changed. `docs/COMMIT_MAP.md` maps old to new. Older docs cite old hashes: `fc2973a` is now `c1916af` (v1 questions), `da8dea5` is now `73ad350` (v2 questions), `83cea5a` is now `af557d3` (dev fixes) and `f3b43ee` is now `b4dd122` (eval-v1).

## 2. Data

- **Sales:** the real UCI "Online Retail II" dataset, a UK online gift retailer from 2009-12-01 to 2011-12-09.
  - 1,067,371 raw rows, cleaned to 1,041,670 sales rows and 19,493 return rows (invoices starting with "C").
  - 236,121 sales rows have no customer id. They are kept for revenue.
  - Loaded into DuckDB as tables `sales` and `returns` plus views `products` and `customers`.
  - `is_merchandise` separates real products from postage and fees.
  - Parquet copies (7.4 MB) are committed in `data/deploy/`, so any host can rebuild the database.
- **Policies:** 6 **synthetic** documents for the fictional "Northwind Gifts": Returns and Refunds, Warranty and Product Quality, Shipping and Delivery, Wholesale and B2B Terms, Loyalty Programme, Customer Data and Privacy.
  - Written in Markdown with numbered clauses and rendered to PDF with reportlab.
  - Ingested from the PDFs into 182 clause-level chunks, each keeping doc, section, clause_id and page.

## 3. Stack (only what the code uses)

| Area | Tools |
|---|---|
| Language | Python 3.11 |
| Orchestration | LangGraph (`StateGraph`, `MemorySaver` per thread) |
| LLM calls | langchain-google-genai, Gemini API, Pydantic structured output |
| Retries | tenacity (exponential backoff with jitter, max 5) |
| Retrieval | sentence-transformers `BAAI/bge-small-en-v1.5`, Chroma, rank_bm25, reciprocal rank fusion (k=60), optional reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| SQL | DuckDB (default), Google BigQuery (optional, free sandbox, google-cloud-bigquery), sqlglot |
| Data loading | pandas, openpyxl, pypdf, reportlab |
| App | FastAPI, uvicorn, Streamlit, httpx |
| Testing and CI | pytest (170 tests), GitHub Actions nightly workflow (passing on GitHub) |
| Hosting and packaging | Streamlit Community Cloud (live demo), Dockerfile (not built yet) |

- **Answering model:** `gemini-3.5-flash-lite` for every component. Thinking level: router low, SQL low, RAG medium, combiner medium.
- **Judge model:** `gemini-3.6-flash` when it had quota, otherwise the answering model (self-judging).
- **Langfuse:** the code supports it but it is not installed and no keys are set, so only local JSONL tracing is used.

## 4. Architecture

```
user -> Streamlit UI or FastAPI POST /ask -> usage limiter -> LangGraph:
  router (Pydantic: route, standalone_question, data_question, policy_question)
    data   -> sql_agent -> combiner -> verifier
    policy -> rag_agent -> combiner -> verifier
    both   -> sql_agent -> rag_agent -> combiner -> verifier
    out_of_scope -> refuse
```

- **Router:** rewrites follow-ups ("And in December?") into standalone questions using the chat history. For mixed questions it gives each agent only its own sub-question.
- **SQL guardrails (all in code):**
  - one `SELECT` only, known tables and columns only
  - file and system functions blocked
  - numeric constants in the top-level select list rejected
  - `LIMIT 200` added, 5 second timeout, read-only connection
  - one repair attempt, with the reason logged to `logs/sql_repairs.jsonl`
- **Warehouse switch:** `RIA_WAREHOUSE=duckdb|bigquery` (DuckDB is the default). The model always writes DuckDB SQL and the same validator checks it. On BigQuery, sqlglot translates the safe query and adds the project and dataset to every table name (`src/ria/warehouse.py`). BigQuery jobs have a 500 MiB maximum-bytes-billed cap and a 15 second timeout, and the agent uses a read-only service account.
- **If the explanation step fails:** the answer is a safe message plus the result rows, marked `explanation_failed`. The error text goes to the trace only.
- **Tracing:**
  - Every model call goes to `logs/llm_calls.jsonl`: node, model, input, output, cached and reasoning tokens, latency, cost, cache hit.
  - Every request goes to `logs/traces.jsonl`: route, trajectory, verified flag, totals.
  - Cost uses `config.PRICES`. These were checked against the official pricing page on 2026-10-25: flash-lite input $0.30, output $2.50, cached $0.03 per million tokens.
- **Local response cache** for development: one flag, off for all final evals.
- **Demo protection:**
  - 5 live questions per visitor and 60 per day for the whole demo (US Pacific quota day).
  - If Gemini reports its daily quota is used up, live questions pause until the reset.
  - 20 pre-computed answers from a real eval-v1 run cost no quota.

## 5. How it was built (6 phases)

1. **Documents and RAG.** Synthetic policies, PDF ingestion, hybrid retrieval, citation format and parser, retrieval eval.
2. **Sales database and SQL agent.** Dataset load and cleaning, sqlglot guardrails, 15-question SQL eval with gold SQL.
3. **LangGraph workflow.** Router, agents, combiner, verifier, refuse node, memory, trajectory logging, dev cache.
4. **Evaluation harness.** 57 questions (20 held-out, 37 dev), gates and observations, LLM judge, human-label template. Held-out questions were committed before any run. Results frozen as eval v1.
5. **Observability and token economics.** Cost per call and per request, token report, caching experiment, explicit-cache trial, nightly workflow.
6. **API, UI, deploy.** FastAPI, Streamlit with a "How this was answered" panel, usage limits, pre-computed examples, Dockerfile, deploy guide, README with a Mermaid diagram.

Later in the session:
- Docs were audited so every number has a command behind it.
- A bug was fixed.
- The repo was pushed to GitHub.
- Eval v2 was run (section 6b).
- **Cloud phase (BigQuery):** a warehouse switch, a loader script and a parity check (section 6c), committed as `7598531`.

## 6. Results (all measured)

**Retrieval** (20 dev policy questions):

| Method | Recall@5 | MRR@5 |
|---|---|---|
| BM25 | 0.70 | 0.608 |
| Vector | 0.95 | 0.800 |
| Hybrid | 0.95 | 0.850 |
| Hybrid + reranker | 0.95 | 0.771 |

- The reranker made things worse, so it is off.
- Dev question p10 is missed by every method.
- A query-rewrite step (fused) reached MRR 0.877. It was not adopted, because a rule set before the run required a gain of at least 0.03.

**Eval v1** (frozen; 20 held-out and 37 dev questions; `gemini-3.5-flash-lite`; cache off; 206 API calls):
- **Held-out:**
  - Router: 20 of 20. Trajectory: 20 of 20 (95% lower bound 84%).
  - SQL execution accuracy: 10 of 10 (95% lower bound 72%).
  - Retrieval recall: 10 of 10. Refusals: 4 of 4.
- **Dev:** router and trajectory 35 of 37 (p07 and p19 policy questions sent to "both"); SQL 15 of 16; recall 20 of 21.
- **Gates (all pass):**
  - No unsupported number or citation: 16 of 16 held-out, 36 of 36 dev.
  - Out-of-scope refused: 4 of 4, 1 of 1.
  - Unsafe SQL blocked: 51 of 51.
  - Gold queries wrongly blocked: 0 of 26.
- **Verifier:** it changed 0 of 52 real answers. Its removal path is shown only by tests and an injected-fault test (it removed a made-up 87% and a fake §9.9 citation).
- **Repeatability:** the same 10 held-out SQL questions scored 10 of 10, 10 of 10 and 9 of 10 in three runs. On h14 the model added `is_merchandise = TRUE` and returned 245,841.79 instead of 679,179.41. The model ignores temperature 0.
- **SQL dev eval** (Phase 2): 14 of 15. The miss (q15) returns quarter labels "Q1" to "Q4" instead of 1 to 4, with identical revenue numbers.

**Cost** (20 held-out questions, production layout; paid-tier list prices; the free tier bills nothing):
- Per 1,000 queries: data $0.827, policy $1.881, both $4.214, out of scope $0.275, overall mix $1.710.
- Output tokens are 79.1% of cost. Thinking tokens are 71.8% of output tokens and 56.8% of cost.
- The schema is 24.3% of input tokens. Retrieved policy context is 11.2%.

**Caching:**
- **Implicit caching did not trigger:** 0 cached tokens in both prompt layouts and in an identical-prompt probe.
- **Explicit caching was refused:** the 879-token SQL prompt is below the 1,024-token minimum, and the free tier has a cache storage limit of 0.
- **Ceiling:** even caching every input token would save at most 18.8%.

**Quota:** the API's own error message states a free limit of 500 requests a day for `gemini-3.5-flash-lite`.

## 6b. Eval v2 (fresh held-out set, run 2026-10-26 after the dev fixes)

- **Setup:** 20 questions committed before the fixes and before any run (commit `73ad350`), run once with the fixed system, cache off, `gemini-3.5-flash-lite`. All 15 answered questions were judged by `gemini-3.6-flash` (no self-judging). 68 API calls.
- **Gates:** all pass. No unsupported content 15 of 15, out-of-scope refused 4 of 4, unsafe SQL blocked 51 of 51.
- **Results:**
  - Router and trajectory: 19 of 20 (95% lower bound 76%).
  - SQL execution accuracy: 10 of 10, including a subquery question.
  - Retrieval recall@5: 7 of 8.
  - "Not covered" answered correctly: 1 of 2.
  - Verifier changed 0 of 15 answers.
- **The misses:**
  - **v11** ("opening hours", not covered by the policies): refused as out of scope instead of answered "not covered".
  - **v09** ("room spray to Canada"): the clause on aerosols was not retrieved, so the answer wrongly said "not covered". The judge still scored it 5 of 5, which shows the judge cannot catch retrieval misses.
- **Cost:** with the 20 eval v1 held-out questions (40 requests in total), $1.713 per 1,000 queries for that mix, and $0.319 to $4.159 by route. The longer router prompt raised the out-of-scope cost.

## 6c. Cloud warehouse: BigQuery (run 2026-10-27)

- **Setup:** Google BigQuery free sandbox (no card). Two service accounts: a loader (BigQuery Data Editor and Job User) and a read-only agent account (BigQuery Data Viewer and Job User). Keys live outside the repo. Their paths and the project id are only in `.env`.
- **Load:** `python scripts/load_bigquery.py` loads the committed parquet files in about 24 seconds and creates the `products` and `customers` views in BigQuery SQL. Row counts match DuckDB: 1,041,670 sales and 19,493 returns.
- **Parity check:** `python scripts/bq_parity.py`, no model calls, BigQuery cache off. 67 queries: gold SQL from the question files, plus SQL the agent wrote in runs `run-phase4` and `run-v2`.
  - Same outcome on DuckDB and BigQuery: **67 of 67**.
  - Matched the gold answer: 66 of 67 on both. The miss is the agent's SQL for dev question q15, which fails the same way on both.
  - Bytes billed per query: median 25.2 MB, max 64.0 MB.
  - Time per query: median 1.52 s on BigQuery against 0.01 s on local DuckDB.
- **Cap check:** a query that needed 63,963,136 bytes was rejected by BigQuery under a 20 MiB cap before it ran.
- **End to end:** the real SQL agent answered one question on BigQuery ("total revenue from Spain in 2010", £41,494.97, matching gold). This is a smoke test, not an eval. A full LLM eval on BigQuery has not been run.
- **Sandbox limit:** tables expire after 60 days (around 2026-12-26). Run the loader again to restore them.
- Full table: section `cloud_bigquery_parity` in `docs/results.md`.

## 7. Problems found and fixed

1. **Invented constants.** The SQL agent put `28 AS return_window_days` in a query. The verifier trusted it because it sat in the SQL rows (the policy says 30). Fixed with a guardrail rejecting numeric constants in the top-level select list (text labels are still allowed), plus the router splitting mixed questions. Covered by tests in `tests/test_phase2.py`, `tests/test_tracing.py` and `tests/test_graph.py`.
2. **Raw API error text became an answer.** When the explanation step hit the quota, the error text reached the user and the verifier stripped pieces of it. Fixed in commit `f3c2bb8`: a safe message plus the rows, an `explanation_failed` flag, and the error in the trace only. Covered by `tests/test_explanation_failure.py`.
3. **List-shaped replies.** Gemini sometimes returns reply content as a list of parts, which crashed the combiner. Fixed so callers always get text, with a test.
4. **Unsupported "for a retail customer".** The RAG prompt made the model add this phrase where nobody asked (13 of 21 dev policy answers in eval v1). Fixed by changing the prompt: 0 of 21 after the fix (`python scripts/doc_facts.py`, block 6).
5. **Two dev misroutes** (p07, p19). The router prompt got disambiguation rules and examples. Dev router accuracy went from 35 of 37 to 37 of 37 (p19 is no longer a clean test, because an example paraphrases it).
6. **Wrong docs claims.** These were removed or corrected: "about 440 calls a day" (the real figure is 500), the v2 commit order, and numbers with no command behind them.
7. **Environment issues.** AVG antivirus intercepts HTTPS on the dev machine, fixed with a CA bundle. The same issue made the first BigQuery load hang silently, because the Google sign-in library retried without printing anything. Setting `REQUESTS_CA_BUNDLE` correctly fixed it. `gemini-2.5-flash` was retired. `gemini-3.8-flash` was slow and quickly ran out of quota, so the project moved to `gemini-3.5-flash-lite`.

## 8. Honesty rules used throughout

- **Traceable numbers:** every number comes from a command. Anything without saved output is marked "found during development, raw output not saved" and carries no number.
- **Held-out first:** held-out questions are committed before any run and are the headline. Dev numbers are optimistic. Nothing is tuned on held-out questions.
- **Eval v1 is frozen:** the code refuses to overwrite its section. Changes are tested on a new held-out set (v2).
- **Strict scoring:** comparisons are never loosened after seeing results (q15 stays a fail).
- **Writing style:** plain language, short sentences, no em dashes.
- **Limits stated openly:** synthetic policies, small eval sets, model nondeterminism, self-judging, a narrow verifier (it checks numbers and citation ids, not meaning, and it trusts SQL rows).

## 9. Key files

- `src/ria/`:
  - `graph.py`, `sql_agent.py`, `warehouse.py`, `rag_agent.py`, `retrieval.py`, `verifier.py`, `llm.py`, `tracing.py`, `limits.py`, `service.py`, `api.py`, `config.py`
  - eval helpers: `eval_metrics.py`, `judge.py`, `token_stats.py`
- `app/streamlit_app.py`: the UI.
- `evals/`:
  - question sets: `questions.jsonl` (v1), `questions_v2.jsonl` (v2), `unsafe_sql.json`
  - scripts: `run_evals.py`, `caching_experiment.py`, `token_report.py`
  - saved runs: `runs/`
- `scripts/doc_facts.py`: prints every derived number. `scripts/sync_readme.py`: copies result tables into the README.
- `scripts/load_bigquery.py`: loads the data into BigQuery. `scripts/bq_parity.py`: runs the same SQL on DuckDB and BigQuery.
- `docs/`:
  - `results.md`: every measured number with its command.
  - `resume_facts.md`: resume-ready facts.
  - `DEPLOY.md`: deploy steps.
  - `COMMIT_MAP.md`: old to new commit hashes.
- `.github/workflows/nightly-eval.yml`: offline tests plus a 10-question eval every night.

## 10. Status: done, pending, planned

**Done:**
- All 6 phases built and committed; 170 tests pass.
- Eval v1 is frozen.
- The docs audit is done.
- The error-text bug is fixed.
- The repo is pushed to GitHub with the owner as author.
- `docs/resume_facts.md` is written.
- A clean clone started locally with `scripts/start.sh`.
- The nightly eval command was run locally: 10 of 10 questions, exit code 0, 27 calls.
- Eval v2 and the dev re-run (section 6b).
- BigQuery warehouse option, data loaded, parity check 67 of 67 (section 6c).
- **Nightly workflow on GitHub:** a manual run on 2026-10-26 passed every step, including the 10-question eval with gates (https://github.com/sravya520/retail-insights-agent/actions/runs/40386120861). The first scheduled run failed only because the API key secret was not set yet.
- **Live demo:** https://retail-insights-agent-ehpjzeospfrryxgbpejme7.streamlit.app/ (Streamlit Community Cloud, separate demo API key). Checked on 2026-10-27: a pre-computed example opens with its full "How this was answered" panel.

**Pending (needed to finish):**
1. **Judge calibration:** label the 20 answers in `evals/human_labels.jsonl` (0 of 20 done), then run `python evals/judge_agreement.py`.

**Optional:**
- Build and test the Docker image (Docker Desktop was not running).
- Connect Langfuse.
- Run the full SQL eval with the LLM on BigQuery (`RIA_WAREHOUSE=bigquery`). It costs Gemini quota; the parity check already covers the warehouse side.

**Planned or discussed:**
- **More cloud.** BigQuery is done (section 6c). Possible next steps: the same warehouse switch for AWS Athena over S3 and Glue (sqlglot supports its dialect), or deploying the container on Cloud Run or AWS. None of this is done yet.
- **Next steps from the README:**
  - lower the thinking level of the RAG and combiner calls, and measure the quality change
  - fix the p10 retrieval miss
  - run each eval several times to measure the spread
  - add human review of answers the verifier changed
  - use a paid key for the public demo

## 11. Claims about the project that are safe to make

- Built a LangGraph multi-agent assistant that answers business questions over retail sales data (guarded text-to-SQL on DuckDB) and company policy documents (RAG with BGE embeddings, Chroma and hybrid BM25 plus semantic search). It routes each question to SQL, RAG or both, keeps conversation memory and cites policy clauses.
- On a held-out set: router and trajectory 20 of 20; SQL 10 of 10 on the frozen run (9 to 10 of 10 across three runs); 51 of 51 unsafe queries blocked by the gates. A deterministic verifier checks numbers and citations. A guardrail was added after the SQL agent passed invented numbers through query constants, and it is covered by tests.
- Retrieval on 20 dev questions: hybrid recall@5 of 0.95 against 0.70 for keyword search. Do not claim hybrid beat vector search (both are 0.95).
- Traced tokens, cost and latency per call. Cost is $0.28 to $4.21 per 1,000 queries by route, at list prices. Output tokens (mostly thinking) are 79% of cost. Added a nightly GitHub Actions eval with gates, which passes on GitHub.
- Added Google BigQuery as an optional warehouse for the SQL agent: the same guardrails, with sqlglot translating DuckDB SQL to BigQuery, a read-only service account and a bytes-billed cap. 67 of 67 queries gave the same outcome on DuckDB and BigQuery, with a median of 25.2 MB billed per query.
- On a second fresh held-out set (v2, run once after the fixes): router and trajectory 19 of 20, SQL 10 of 10, retrieval recall 7 of 8. All gates passed, and the judge was a different model from the answering model.
- Deployed a public demo on Streamlit Community Cloud (https://retail-insights-agent-ehpjzeospfrryxgbpejme7.streamlit.app/) with usage limits that protect the free model quota and 20 pre-computed examples that cost nothing.
- **Never claim:**
  - caching savings
  - judge calibration
  - cloud deployment of the app (not done yet)
  - a full LLM eval on BigQuery (only the parity check and one smoke test were run)
  - AWS or Athena (not done)
  - that hybrid search beat vector search
