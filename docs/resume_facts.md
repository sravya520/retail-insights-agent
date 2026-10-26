# Resume facts: Retail Insights Agent

Every line below comes from `docs/results.md`, from a file saved by a run, or from `python scripts/doc_facts.py` (which reads only saved run files and makes no API calls). For each number the table gives what it measures, what it was measured on, and where it comes from. "NOT MEASURED" means there is no number yet. Commands marked "local only" read files that are git-ignored (`logs/`, `data/db/`), so they work on my machine but not in a fresh clone.

All model answers were produced by `gemini-3.5-flash-lite` unless a line says otherwise. The policy documents are synthetic.

## 1. What the project is

A business user asks questions in plain English about a UK online gift retailer. A LangGraph workflow sends each question to one of four paths: a SQL agent for questions about the sales data, a retrieval (RAG) agent for questions about the company's policy documents, both agents for mixed questions, or a polite refusal for anything else. The SQL agent writes a read-only query that code checks before it runs. The RAG agent answers only from retrieved policy clauses and cites each one. A verifier written in plain code removes any sentence whose number or citation it cannot trace to the SQL result or a retrieved clause. The sales data is the real UCI Online Retail II dataset. The six policy documents are synthetic, written for this project. A FastAPI service and a Streamlit chat UI show each answer with its route, agent steps, SQL, citations, verified flag, tokens, cost and latency.

## 2. Stack actually used

Versions are the ones installed in my environment (command: `python -c "import importlib.metadata as m; ..."`, run on 2026-10-25). `requirements.txt` only sets minimum versions.

| Library | Version | Used for | Where in the repo |
|---|---|---|---|
| langgraph | 1.2.17 | Agent workflow (`StateGraph`), memory per thread (`MemorySaver`) | `src/ria/graph.py` |
| langchain-core, langchain-google-genai | 1.7.0, 4.4.2 | Gemini chat calls, structured output | `src/ria/llm.py` |
| google-genai | 2.33.0 | Token counting and the explicit caching trial | `evals/token_report.py`, `evals/explicit_cache_trial.py` |
| pydantic | 2.14.1 | Router output schema, judge schema, API models | `graph.py`, `judge.py`, `api.py` |
| tenacity | 9.1.4 | Retry with exponential backoff and jitter on 429 and 5xx | `src/ria/llm.py` |
| sentence-transformers (with torch 2.14.1) | 6.2.0 | Embeddings `BAAI/bge-small-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` | `ingest_docs.py`, `retrieval.py` |
| chromadb | 1.5.9 | Vector store | `ingest_docs.py`, `retrieval.py` |
| rank_bm25 | 0.2.2 | BM25 keyword search | `retrieval.py` |
| pypdf, reportlab | 6.21.2, 5.1.0 | Read and write the policy PDFs | `ingest_docs.py`, `make_policy_pdfs.py` |
| duckdb | 2.0.0 | Sales database, read-only queries | `sql_db.py`, `sql_agent.py` |
| sqlglot | 30.24.0 | SQL parsing and guardrails, DuckDB to BigQuery translation | `sql_agent.py`, `warehouse.py` |
| google-cloud-bigquery | 3.46.2 | Optional BigQuery warehouse (free sandbox) | `warehouse.py`, `scripts/load_bigquery.py` |
| pandas, openpyxl | 3.1.0, 3.1.5 | Load and clean the Excel dataset | `sql_db.py` |
| fastapi, uvicorn | 0.162.0, 0.54.0 | REST API | `src/ria/api.py` |
| streamlit | 1.66.1 | Chat UI | `app/streamlit_app.py` |
| httpx | 0.28.1 | UI to API calls | `src/ria/ui_backend.py` |
| python-dotenv, tzdata | 1.2.4, 2026.5 | Secrets from `.env`, US Pacific quota day | `config.py`, `limits.py` |
| pytest | 9.1.1 | 170 offline tests | `tests/` |

Listed but not used: `requests` is in `requirements.txt` but no repository code imports it. `numpy` is not imported directly. `langfuse` is referenced in `tracing.py` but is not installed, so it never ran. Docker (`Dockerfile`) exists as a file but was never built (see section 11). The GitHub Actions workflow has run on GitHub (section 10).

## 3. Data

| Fact | Number | What it measures | Source |
|---|---|---|---|
| Raw dataset rows | 1,067,371 | All rows in both sheets of UCI Online Retail II | `data/db/load_report.json`, from `python -I -m ria.sql_db` (local only) |
| Clean sales rows | 1,041,670 | Non-cancelled lines with quantity > 0 and price > 0 | same |
| Return rows | 19,493 | Cancelled invoice lines (invoice starts with "C") | same |
| Sales rows without a customer ID | 236,121 | Kept, so they count in revenue | same |
| Date range | 2009-12-01 to 2011-12-09 | First and last invoice date | same |
| Policy documents | 6 | Synthetic policies (Markdown and PDF) | `data/docs/` |
| Policy chunks | 182 | One chunk per numbered clause | output of `python -m ria.ingest_docs`; `data/chroma/chunks.json` has 182 entries |
| Eval v1 held-out questions | 20 | 6 data, 6 policy, 4 both, 4 out of scope. Committed before any run (`fc2973a`) | `evals/questions.jsonl` |
| Eval v1 dev questions | 37 | 15 data, 20 policy, 1 both, 1 out of scope. Used while building, so optimistic | `evals/questions.jsonl` |
| Eval v2 held-out questions | 20 | 6 data, 6 policy, 4 both, 4 out of scope. Committed before any run (`da8dea5`). Not run yet | `evals/questions_v2.jsonl` |
| Retrieval eval questions | 20 | Dev policy questions p01 to p20 | `evals/policy_questions.jsonl` |
| SQL eval questions | 15 | Dev data questions q01 to q15 | `evals/sql_questions.jsonl` |
| Unsafe SQL test queries | 51 | Queries the guardrail must block | `evals/unsafe_sql.json` |

## 4. Retrieval

Measured on the 20 **dev** policy questions (p01 to p20). Command: `python evals/retrieval_eval.py` (no Gemini calls). Source: section "Retrieval comparison" in `docs/results.md`. Recall@5 = gold clause in the top 5. MRR@5 = reciprocal rank of the first gold clause in the top 5.

| Method | Recall@5 | MRR@5 |
|---|---|---|
| BM25 only | 0.70 | 0.608 |
| Vector only (bge-small) | 0.95 | 0.800 |
| Hybrid, BM25 + vector with RRF (k=60) | 0.95 | 0.850 |
| Hybrid + cross-encoder reranker (top 20 reranked to top 5) | 0.95 | 0.771 |
| Hybrid + reranker, pool of 40 | 0.95 | 0.762 |

- The reranker lowered MRR on this set, so it is off in production. The one question every method misses is p10.
- End to end on the **held-out** v1 questions, a gold clause was among the 5 retrieved clauses for 10 of 10 policy and mixed questions (`python evals/run_evals.py`, section "Eval v1").
- End to end on the **held-out v2** questions (run once on 2026-10-26 after the dev fixes), a gold clause was retrieved for 7 of 8 policy and mixed questions. The miss is v09 ("room spray" against the clause on "aerosols"), and its answer wrongly said "not covered". Source: section "Evaluation eval_v2" in `docs/results.md` (`python evals/run_evals.py --questions evals/questions_v2.jsonl ...`).
- A query-rewrite step (original and rewritten query fused) gave recall 0.95 and MRR 0.877 on the same 20 dev questions. It was not adopted, because a rule set before the run required a gain of at least 0.03 (`python evals/rewrite_experiment.py`).

## 5. SQL agent

| Fact | Number | Measured on | Source |
|---|---|---|---|
| Execution accuracy, held-out | **10 of 10** | The 10 eval v1 held-out questions that have a gold SQL answer (6 data, 4 both). Agent result must match gold within 0.5% | `python evals/run_evals.py`, run `run-phase4`, section "Eval v1" |
| 95% lower bound for that score | 72% | Wilson interval for 10 of 10 | same report |
| Same 10 questions, two later runs | 10 of 10, then **9 of 10** | Runs `caching-question_first` and `caching-stable_first`. In the second, question h14 added a filter `is_merchandise = TRUE` and returned 245,841.79 instead of the gold 679,179.41 | `python scripts/doc_facts.py`, block 1 |
| Execution accuracy, dev | 14 of 15 | Dev data questions q01 to q15. The miss (q15) returns quarter labels "Q1" to "Q4" instead of 1 to 4. The revenue numbers are identical | `python evals/sql_eval.py`, section "SQL agent execution accuracy" |
| Execution accuracy, held-out v2 | **10 of 10** | The 10 eval v2 held-out questions with a gold SQL answer, including one that needs a subquery. One run, after the dev fixes | section "Evaluation eval_v2" in `docs/results.md` (`python evals/run_evals.py --questions evals/questions_v2.jsonl ...`) |
| Execution accuracy, dev, in eval v1 | 15 of 16 | Dev data questions plus the dev mixed question | `python evals/run_evals.py`, section "Eval v1" |
| Warehouse parity, DuckDB against BigQuery | **67 of 67** same outcome | 67 queries: gold SQL plus agent SQL replayed from runs `run-phase4` and `run-v2`. 66 of 67 matched gold on both (q15 fails on both). No model calls | section "cloud_bigquery_parity" in `docs/results.md` (`python scripts/bq_parity.py`) |
| BigQuery bytes billed per query | median 25.2 MB, max 64.0 MB | Same 67 queries, BigQuery query cache off. A 20 MiB cap rejected the 64 MB query before it ran | section "cloud_bigquery_parity" in `docs/results.md` (`python scripts/bq_parity.py`) |
| Repairs needed in eval v1 | 1 (h04) | The first query failed and the one repair passed. The reason was not logged | `python scripts/doc_facts.py`, block 5 |

## 6. Router and trajectory

| Fact | Number | Measured on | Source |
|---|---|---|---|
| Router accuracy, held-out | **20 of 20** (95% lower bound 84%) | Eval v1 held-out questions | `python evals/run_evals.py`, section "Eval v1" |
| Trajectory accuracy (exact sequence of agent steps), held-out | **20 of 20** (95% lower bound 84%) | same | same |
| Router and trajectory accuracy, dev | 35 of 37 | Eval v1 dev questions. Both misses (p07, p19) were policy questions sent to the mixed path | section "Eval v1"; the ids from `python scripts/doc_facts.py`, block 5 |
| Refusal of out-of-scope questions, held-out | 4 of 4 | same | same |
| Router and trajectory accuracy, held-out v2 | 19 of 20 (95% lower bound 76%) | Eval v2 held-out questions, one run after the dev fixes. The miss (v11, "opening hours", not covered by the policies) was refused as out of scope instead of answered "not covered" | section "Evaluation eval_v2" in `docs/results.md` (`python evals/run_evals.py --questions evals/questions_v2.jsonl ...`) |
| Router and trajectory accuracy, dev, after the fixes | 37 of 37 (was 35 of 37) | Eval v1 dev questions. p19 is no longer a clean test, because a router example paraphrases it | section "Evaluation dev_after_fixes" |

## 7. Verifier and guardrails

What the verifier blocks: a sentence with a number that is not in the SQL result, the user's question, or the clause the sentence cites, and a sentence with a citation that matches no retrieved clause. What the SQL guardrail blocks: anything that is not a single `SELECT`, unknown tables or columns, file and system functions, and numeric constants in the top-level select list. It also adds `LIMIT 200` and runs read-only with a 5 second timeout.

| Fact | Number | Measured on | Source |
|---|---|---|---|
| Gate 1: no unsupported number or citation in a final answer | Pass. Held-out 16 of 16, dev 36 of 36 | Every answered eval v1 question, verifier run again on the final answer | `python evals/run_evals.py`, section "Eval v1" |
| Gate 2: out-of-scope questions refused | Pass. Held-out 4 of 4, dev 1 of 1 | same | same |
| Gate 3: unsafe SQL blocked | Pass. 51 of 51 | `evals/unsafe_sql.json`, no model involved | same |
| Gold queries wrongly blocked | 0 of 26 | All eval v1 gold SQL queries | same |
| Real answers the verifier changed | 0 of 52 | All answered eval v1 questions | same |
| Injected-fault test | 2 of 2 fake sentences removed | One made-up number (87%) and one fake citation (§9.9) added to a real answer | `python evals/graph_examples.py`, section "Graph example runs" |

The bug found and fixed (invented constants). Found during development, raw output not saved. Found and fixed, covered by tests. In a Phase 3 example run the SQL agent added a constant column, `28 AS return_window_days`, to its query. The 28 matched no data (the policy says 30 days), and the verifier accepted it because it sat in the SQL result. Two fixes followed. A guardrail rejects numeric constants in the top-level select list (text labels like `'Germany' AS country` are still allowed). The router splits a mixed question so each agent sees only its own part. Tests: `tests/test_phase2.py` rejects `SELECT COUNT(*) AS n, 28 AS return_window_days FROM returns`, `tests/test_tracing.py` checks that a numeric constant triggers a logged repair, and `tests/test_graph.py` checks the split. Source: Phase 3 section of `docs/results.md`.

A second bug found and fixed (error text as an answer). When the SQL agent's explanation step failed, the raw API error text became the answer (seen on q10 in run `run-dev-after-fixes`). Now the answer is a safe message plus the SQL rows, the run is marked `explanation_failed`, and the error text goes to the trace log only. Covered by `tests/test_explanation_failure.py`, which also checks that the verifier never sees the error text. Fixed in commit `388a006`.

## 8. Cost and tokens

Measured on the 20 eval v1 held-out questions, production prompt layout, cache off, run `caching-stable_first`. Prices are the paid-tier list prices in `config.PRICES` (checked against the official pricing page on 2026-10-25). The free tier bills nothing. Command: `python evals/token_report.py`, section "Token economics".

| Route | Requests | Cost per 1,000 queries |
|---|---|---|
| Data | 6 | $0.827 |
| Policy | 6 | $1.881 |
| Both | 4 | $4.214 |
| Out of scope | 4 | $0.275 |
| All 20 (this mix) | 20 | $1.710 |

Combined with the 20 eval v2 questions (40 requests), the cost is $1.713 per 1,000 queries for that mix, and $0.319 to $4.159 by route (section "phase6_tokens_v1v2" in `docs/results.md`, `python evals/token_report.py --records ... --section phase6_tokens_v1v2`).

| Fact | Number | Measured on | Source |
|---|---|---|---|
| Share of cost from output tokens | 79.1% | Same 20 requests (23,839 input, 10,823 output tokens) | `python scripts/doc_facts.py`, block 2 |
| Thinking (reasoning) tokens | 7,771 of 10,823 output tokens (71.8%), 56.8% of cost | The 54 model calls of the same run | `python scripts/doc_facts.py`, block 4 (local only: reads `logs/`) |
| Most cost saved if every input token were cached | 18.8% | Same 20 requests | `python scripts/doc_facts.py`, block 2 |
| Schema description share of input tokens | 24.3% | Same 20 requests | `python evals/token_report.py` |
| Retrieved policy context share of input tokens | 11.2% | Same 20 requests | same |
| Prompt caching | **Did not trigger.** 0 cached tokens in both prompt layouts (23,882 and 23,839 input tokens) and 0 when the same 894-token prompt was sent 4 times in a row | Same 20 questions in two layouts | `python evals/caching_experiment.py` |
| Explicit context caching | **Refused.** The SQL prompt (879 tokens) is under the 1,024-token minimum, and the free tier has a cache storage limit of 0 | SQL system prompt | `python evals/explicit_cache_trial.py` |

## 9. Judge calibration

**Pending human labels.** `evals/human_labels.jsonl` has 20 answers and 0 of them are labelled. Judge agreement is NOT MEASURED. Do not use judge scores on a resume. In eval v1 most scores were self-judged. In eval v2 all 15 were judged by `gemini-3.6-flash`, but it still gave 5 of 5 to the wrong v09 answer, because it checks answers only against the retrieved evidence. It cannot catch retrieval misses.

## 10. CI (nightly workflow)

| Fact | Status | Source |
|---|---|---|
| Nightly workflow on GitHub (`.github/workflows/nightly-eval.yml`, scheduled and manual) | **Passed** on a manual run on 2026-10-26 (commit `5346e5a`): offline gate tests and the 10-question eval with gates, all steps green. The first scheduled run failed before the `GOOGLE_API_KEY` secret was added | GitHub Actions run 40386120861 (https://github.com/sravya520/retail-insights-agent/actions/runs/40386120861) |
| The workflow's eval command run locally | 10 of 10 questions finished, all three gates passed, exit code 0, 27 model calls | Phase 5 section of `docs/results.md`; call count from `evals/runs/nightly-local/meta.json`, local only |
| Exit codes (0 pass, 1 gate failed, 2 run stopped early) | Tested: 3 tests pass | `tests/test_nightly_gate.py`, part of `pytest` |
| Offline test suite | 170 passed | `pytest`, run on 2026-10-27 |

## 11. Deployment status and links

| Item | Status |
|---|---|
| GitHub URL | https://github.com/sravya520/retail-insights-agent |
| Demo URL | https://retail-insights-agent-ehpjzeospfrryxgbpejme7.streamlit.app/ (Streamlit Community Cloud). Checked on 2026-10-27: the page loads and a pre-computed example shows the answer, route, trajectory, SQL, verified flag, tokens, cost and latency |
| Docker image | Not built. The Dockerfile is untested |
| What was tested instead | A clean clone started with `scripts/start.sh` rebuilt the database and the vector index from committed files, and the API and UI health checks passed. Found during development, raw output not saved (Phase 6 section of `docs/results.md`) |

## 12. Limits and known weaknesses

- The six policy documents are synthetic. The sales data covers December 2009 to December 2011 only.
- The eval sets are small: 20 held-out and 37 dev questions in v1. A perfect 20 of 20 has a 95% lower bound of 84%. Route groups in the cost report hold 4 to 6 questions.
- The model is not deterministic. It ignores temperature 0. Held-out question h14 gave the gold answer in two runs and a different answer in a third (section 5).
- **The one wrong pre-computed example.** Of the 20 pre-computed examples in the demo (`data/examples/examples_v1.json`, built by `python scripts/build_examples.py` from run `caching-stable_first`), 15 match the gold answer, 1 does not, and 4 are refusals with no gold check. The wrong one is h14: "What was the total value of returns in 2010, ...". The answer says £245,841.79 because the query added `is_merchandise = TRUE`. The gold answer is £679,179.41. It is kept and labelled as wrong in the UI.
- The verifier checks numbers and citation ids only, not meaning. It trusts the SQL rows. It ignores numbers written as words.
- Retrieval misses dev question p10 with every method.
- The router sent 2 dev policy questions (p07, p19) to the mixed path in eval v1. After the prompt fix, dev router accuracy is 37 of 37 (p19 is no longer a clean test). On the held-out v2 set the router refused v11, which the policies do not cover, as out of scope.
- The RAG prompt in eval v1 added "for a retail customer" when the question did not mention retail in 13 of 21 dev policy answers. After the prompt fix: 0 of 21 (`python scripts/doc_facts.py`, block 6).
- Free-tier quota: the API's own quota error for `gemini-3.5-flash-lite` states a limit of 500 requests per day (saved in `evals/runs/run-dev-after-fixes/records.jsonl`; `python scripts/doc_facts.py`, block 3).
- The partial dev re-run (`run-dev-after-fixes`) shows "Answers the verifier changed 1/10". That was not the verifier catching a model mistake: on q10 the raw quota error text became the answer and the verifier removed its 3 sentences with numbers. The bug is now fixed (section 7), but the saved record of that run still shows the old behaviour. Its SQL rows were correct, so the run's SQL accuracy counts q10 as correct.
- If the explanation step fails now, the user gets the result rows without a written explanation.
- Memory: on my machine the agent came close to the documented minimum memory of Streamlit Community Cloud. Found during development, raw output not saved.

## 13. Eval v2

**Done (2026-10-26).** 20 fresh held-out questions, committed before the fixes and before any run (`73ad350`, old hash `da8dea5`), run once with the fixed system. All gates passed. Router and trajectory 19 of 20, SQL 10 of 10, retrieval recall 7 of 8, refusals 4 of 4, "not covered" answered correctly 1 of 2, verifier changed 0 of 15. All 15 answers were judged by `gemini-3.6-flash` (no self-judging). Source: section "Evaluation eval_v2" in `docs/results.md` (`python evals/run_evals.py --questions evals/questions_v2.jsonl ...`).

## How the 24 untraceable claims were resolved

The first version of this file listed 24 statements in `docs/results.md` and `README.md` that had no command behind them. Commit `0cd31c5` fixed them. "Rewritten" means the statement now says "found during development, raw output not saved" and keeps no number. "Command" means the number is now printed by `python scripts/doc_facts.py`.

| # | Claim | Outcome |
|---|---|---|
| 1 | SQL agent wrote `28 AS return_window_days` | Rewritten. Kept as "found and fixed, covered by tests" (tests named in section 7) |
| 2 | First RAG example run said "not covered" for Spain, reranker detail | Rewritten, reranker detail removed |
| 3 | Default thinking level gave "not covered" | Rewritten |
| 4 | `gemini-2.5-flash` 404, `gemini-3.8-flash` 20 per day and 224 to 327 s | Rewritten, numbers removed |
| 5 | First SQL run "about 2 s per question", "13 of 13 answered passed" | Rewritten, numbers removed |
| 6 | Smoke test "8 more calls" | Number removed |
| 7 | PDFs "2 to 3 pages each" | Deleted |
| 8 | "23 model calls (10 router, 8 SQL, 3 RAG, 2 combiner)" | Rewritten, numbers removed |
| 9 | Guardrail once blocked `'Germany' AS country` | Rewritten. Current behaviour is covered by tests |
| 10 | Ids of judge scores below 5 and the p07 answer | Command (block 5) |
| 11 | Input costs, output tokens, 22.8 s outlier, 79%, 19% | Output tokens and 79.1% and 18.8% now by command (block 2). Input costs and the outlier removed |
| 12 | "No SQL repairs in 50 requests" | Deleted |
| 13 | "About 440 calls a day", "`gemini-3.6-flash` allowed 17" | Deleted. Replaced by the 500 per day limit from the API error (block 3) |
| 14 | Phase 6 one-off checks (prices, clean checkout, 502 and 503, memory, import time, browser) | Rewritten, all numbers removed. Tests named where they exist |
| 15 | Docs minimum of 4,096 tokens | Removed. The 1,024-token minimum reported by the API (explicit cache trial) is kept |
| 16 | "Phase 4 will rerun every component" | Rewritten: Phase 4 did this |
| 17 | "Retail is assumed ... says so once" | Rewritten: that rule was replaced after eval v1 |
| 18 | Prices "read through a page summariser" | Rewritten |
| 19 | "Its daily limit is not shown to me" | Replaced by the 500 per day limit (block 3) |
| 20 | "Those 10 say nothing about the fixes" | Rewritten: all 10 were routed correctly, none tests the policy fixes, q10 exposed a bug |
| 21 | Confusing partial-run header and "verifier changed 1/10" | Explained in a note under that run's table |
| 22 | README: v2 committed "after" the fixes | Corrected: `da8dea5` comes before the fix commit `83cea5a` |
| 23 | README: "every number is copied by sync_readme.py" and hand-typed prose numbers | Rewritten. Prose numbers now come from the synced tables or the command. 440, 700 MB, 130 MB, 10 to 25 s and "about 85%" removed |
| 24 | README: "Live demo answers take roughly 10 to 25 seconds" | Deleted |

Also added to `docs/results.md`, under the eval v1 block: "SQL execution accuracy, held-out: 10 of 10 on eval v1; two later runs on the same questions gave 10 of 10 and 9 of 10" (command: block 1). The frozen eval v1 block itself was not changed.

What still has no repository command, by design: the statements marked "found during development, raw output not saved". They carry no numbers. The nightly dry run in the Phase 5 section of `docs/results.md` (10 of 10 questions, 27 calls) comes from a run folder that is git-ignored (`evals/runs/nightly-local/`), so it is local only.
