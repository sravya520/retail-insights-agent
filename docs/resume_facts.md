# Resume facts: Retail Insights Agent

Every line below comes from `docs/results.md`, from a file saved by a run, or from a command I ran on 2026-10-25. For each number the table gives what it measures, what it was measured on, and where it comes from. "NOT MEASURED" means there is no number yet. Commands marked "local only" read files that are git-ignored (`logs/`, `data/db/`), so they work on my machine but not in a fresh clone.

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
| sqlglot | 30.24.0 | SQL parsing and guardrails | `sql_agent.py` |
| pandas, openpyxl | 3.1.0, 3.1.5 | Load and clean the Excel dataset | `sql_db.py` |
| fastapi, uvicorn | 0.162.0, 0.54.0 | REST API | `src/ria/api.py` |
| streamlit | 1.66.1 | Chat UI | `app/streamlit_app.py` |
| httpx | 0.28.1 | UI to API calls | `src/ria/ui_backend.py` |
| python-dotenv, tzdata | 1.2.4, 2026.5 | Secrets from `.env`, US Pacific quota day | `config.py`, `limits.py` |
| pytest | 9.1.1 | 155 offline tests | `tests/` |

Listed but not used: `requests` is in `requirements.txt` but no repository code imports it. `numpy` is not imported directly. `langfuse` is referenced in `tracing.py` but is not installed, so it never ran. GitHub Actions (`.github/workflows/nightly-eval.yml`) and Docker (`Dockerfile`) exist as files but never ran (see sections 10 and 11).

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
- A query-rewrite step (original and rewritten query fused) gave recall 0.95 and MRR 0.877 on the same 20 dev questions. It was not adopted, because a rule set before the run required a gain of at least 0.03 (`python evals/rewrite_experiment.py`).

## 5. SQL agent

| Fact | Number | Measured on | Source |
|---|---|---|---|
| Execution accuracy, held-out | **10 of 10** | The 10 eval v1 held-out questions that have a gold SQL answer (6 data, 4 both). Agent result must match gold within 0.5% | `python evals/run_evals.py`, run `run-phase4`, section "Eval v1" |
| 95% lower bound for that score | 72% | Wilson interval for 10 of 10 | same report |
| Same 10 questions, two later runs | 10 of 10, then **9 of 10** | Runs `caching-question_first` and `caching-stable_first`. In the second, question h14 added a filter `is_merchandise = TRUE` and returned 245,841.79 instead of the gold 679,179.41 | saved records `evals/runs/caching-*.jsonl`, checked by a command I ran |
| Execution accuracy, dev | 14 of 15 | Dev data questions q01 to q15. The miss (q15) returns quarter labels "Q1" to "Q4" instead of 1 to 4. The revenue numbers are identical | `python evals/sql_eval.py`, section "SQL agent execution accuracy" |
| Execution accuracy, dev, in eval v1 | 15 of 16 | Dev data questions plus the dev mixed question | `python evals/run_evals.py`, section "Eval v1" |
| Repairs needed in eval v1 | 1 (h04) | The first query failed and the one repair passed. The reason was not logged | same |

## 6. Router and trajectory

| Fact | Number | Measured on | Source |
|---|---|---|---|
| Router accuracy, held-out | **20 of 20** (95% lower bound 84%) | Eval v1 held-out questions | `python evals/run_evals.py`, section "Eval v1" |
| Trajectory accuracy (exact sequence of agent steps), held-out | **20 of 20** (95% lower bound 84%) | same | same |
| Router and trajectory accuracy, dev | 35 of 37 | Eval v1 dev questions. Both misses (p07, p19) were policy questions sent to the mixed path | same |
| Refusal of out-of-scope questions, held-out | 4 of 4 | same | same |

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

The bug found and fixed (invented constants): in a Phase 3 example run, the SQL agent wrote `SELECT COUNT(...), 28 AS return_window_days ...`. The 28 matched no data (the policy says 30 days), and the verifier accepted it because it sat in the SQL result. Two fixes followed. A guardrail now rejects numeric constants in the top-level select list (it was later narrowed so text labels like `'Germany' AS country` are still allowed). The router now splits a mixed question so each agent sees only its own part. Source: Phase 3 section of `docs/results.md`. The fix is covered by tests (`tests/test_phase2.py`, `tests/test_tracing.py`). The raw output of the run that showed the bug was not kept (see the last section).

## 8. Cost and tokens

Measured on the 20 eval v1 held-out questions, production prompt layout, cache off, run `caching-stable_first`. Prices are the paid-tier list prices in `config.PRICES` (checked against the official pricing page on 2026-10-25). The free tier bills nothing. Command: `python evals/token_report.py`, section "Token economics".

| Route | Requests | Cost per 1,000 queries |
|---|---|---|
| Data | 6 | $0.827 |
| Policy | 6 | $1.881 |
| Both | 4 | $4.214 |
| Out of scope | 4 | $0.275 |
| All 20 (this mix) | 20 | $1.710 |

| Fact | Number | Measured on | Source |
|---|---|---|---|
| Share of cost from output tokens | 79.1% | Same 20 requests (23,839 input, 10,823 output tokens) | one-line command over `evals/runs/caching-stable_first.jsonl` and the prices, run on 2026-10-25 |
| Thinking (reasoning) tokens | 7,771 of 10,823 output tokens (71.8%), 56.8% of cost | The 54 model calls of the same run | one-line command over `logs/llm_calls.jsonl` and `logs/traces.jsonl` (local only) |
| Most cost saved if every input token were cached | 18.8% | Same 20 requests | same command as the 79.1% line |
| Schema description share of input tokens | 24.3% | Same 20 requests | `python evals/token_report.py` |
| Retrieved policy context share of input tokens | 11.2% | Same 20 requests | same |
| Prompt caching | **Did not trigger.** 0 cached tokens in both prompt layouts (23,882 and 23,839 input tokens) and 0 when the same 894-token prompt was sent 4 times in a row | Same 20 questions in two layouts | `python evals/caching_experiment.py` |
| Explicit context caching | **Refused.** The SQL prompt (879 tokens) is under the 1,024-token minimum, and the free tier has a cache storage limit of 0 | SQL system prompt | `python evals/explicit_cache_trial.py` |

## 9. Judge calibration

**Pending human labels.** `evals/human_labels.jsonl` has 20 answers and 0 of them are labelled. Judge agreement is NOT MEASURED. The judge scores in eval v1 are uncalibrated. Most of them were self-judged (the answering model scored its own answers), and the judge gave relevance 5 of 5 to every answer. Do not use the judge scores on a resume.

## 10. CI (nightly workflow)

| Fact | Status | Source |
|---|---|---|
| Nightly workflow on GitHub (`.github/workflows/nightly-eval.yml`, scheduled and manual) | **Untested.** It has never run on GitHub, because the repository is not on GitHub | `git remote -v` prints nothing |
| The workflow's eval command run locally | 10 of 10 questions finished, all three gates passed, exit code 0, 30 model calls | Phase 5 section of `docs/results.md` |
| Exit codes (0 pass, 1 gate failed, 2 run stopped early) | Tested: 3 tests pass | `tests/test_nightly_gate.py`, part of `pytest` |
| Offline test suite | 155 passed | `pytest`, run on 2026-10-25 |

## 11. Deployment status and links

| Item | Status |
|---|---|
| GitHub URL | **None.** No remote is set up for this folder (`git remote -v` prints nothing) |
| Demo URL | **Not deployed** |
| Docker image | Not built. The Dockerfile is untested |
| What was tested instead | A clean clone started with `scripts/start.sh` rebuilt the database and the vector index from committed files in about 40 seconds. Health checks for the API and the UI passed. Source: Phase 6 section of `docs/results.md` |

## 12. Limits and known weaknesses

- The six policy documents are synthetic. The sales data covers December 2009 to December 2011 only.
- The eval sets are small: 20 held-out and 37 dev questions in v1. A perfect 20 of 20 has a 95% lower bound of 84%. Route groups in the cost report hold 4 to 6 questions.
- The model is not deterministic. It ignores temperature 0. Held-out question h14 gave the gold answer in two runs and a different answer in a third (section 5).
- **The one wrong pre-computed example.** Of the 20 pre-computed examples in the demo (`data/examples/examples_v1.json`, built by `python scripts/build_examples.py` from run `caching-stable_first`), 15 match the gold answer, 1 does not, and 4 are refusals with no gold check. The wrong one is h14: "What was the total value of returns in 2010, ...". The answer says £245,841.79 because the query added `is_merchandise = TRUE`. The gold answer is £679,179.41. It is kept and labelled as wrong in the UI.
- The verifier checks numbers and citation ids only, not meaning. It trusts the SQL rows. It ignores numbers written as words.
- Retrieval misses dev question p10 with every method.
- The router sent 2 dev policy questions (p07, p19) to the mixed path. A prompt fix exists, but its effect is NOT MEASURED.
- The RAG prompt in eval v1 added "for a retail customer" to answers where nobody asked (the judge flagged h11, p14 and p15). A prompt fix exists, but its effect is NOT MEASURED.
- Free-tier quota: on 2026-10-25 the API's own quota error for `gemini-3.5-flash-lite` stated a limit of 500 requests per day (saved in `evals/runs/run-dev-after-fixes/records.jsonl`, question q10).
- **Bug found while preparing this file (not fixed, because this task changes no code).** When the SQL agent's explanation step fails, the raw API error text becomes the answer. In the partial dev re-run (question q10, run `run-dev-after-fixes`), the quota error text reached the final answer, and the verifier removed 3 sentences that contained numbers from the error message (429, 500, 3.5). That is why that run's table shows "Answers the verifier changed 1/10". It was not the verifier catching a model mistake. The SQL rows for q10 were still correct, so the run's SQL accuracy counts it as correct.
- Memory: about 624 MB for the agent after the first policy search, plus about 71 MB for an idle Streamlit server (Phase 6 section of `docs/results.md`). That is close to the documented minimum of Streamlit Community Cloud.

## 13. Eval v2

**Pending. NOT MEASURED.** The 20 v2 questions were committed in `da8dea5` before any fix and before any run. The run has not happened, because the free daily quota ran out. The dev re-run after the fixes also stopped after 10 of 37 questions (all data questions), so it does not measure the fixes.

## Claims I could not trace to a command

These are statements in `docs/results.md` or `README.md` that have no command in the repository behind them, or that are stale or wrong. None of them is used in the tables above, except where a table says "command I ran".

**From runs whose raw output was not kept.** The only record of these is my own text, written when I saw the output. The output itself was overwritten by a later run, or exists only in a console log in my temporary session folder.

1. results.md, Phase 3: the SQL agent wrote `28 AS return_window_days`. The run's output was overwritten by the next run. Git history only has my description of it.
2. results.md, Phase 1: the first RAG example run answered "not covered" for the Spain question, and the reranker pushed Shipping §3.2 out of the top 5. That first run was not kept.
3. results.md, "Model, quota and latency": at the default thinking level, the model wrongly said "not covered" for the wholesale question. Seen once, output not kept.
4. results.md: `gemini-2.5-flash` returned a 404, and `gemini-3.8-flash` allowed 20 requests per day. These come from error messages in the session, not from a saved result. Partial trace: `logs/llm_calls_old_pre_fix.jsonl` (local only) has the slow 3.8-flash calls (224 to 327 seconds).
5. results.md: "The first SQL eval run took about 2 seconds per question" and "the first full run hit the 15 requests per minute limit on the last two questions (13 of 13 answered passed)". These come from console logs of runs that were later replaced.
6. results.md: "A smoke test with 2 questions used 8 more calls". Console log only, and the smoke run folder was deleted.

**From one-off commands in my session folder, not scripts in the repository.**

7. results.md, Phase 1: the PDFs are "2 to 3 pages each" (a one-off page count with pypdf).
8. results.md, Phase 3: "23 model calls in total (10 router, 8 SQL, 3 RAG, 2 combiner)" (a one-off count over `logs/llm_calls.jsonl`).
9. results.md, Phase 4: "The rule did block a plain string label such as `SELECT 'Germany' AS country`" (a one-off guardrail check). The current behaviour is covered by tests.
10. results.md, Phase 4: the ids behind the judge's scores below 5 (h11, p14, p15 and q04) and the p07 answer text. These are one-off inspections of `evals/runs/run-phase4/records.jsonl`. The file supports them, but no repository command prints them.
11. results.md, Phase 5: input cost $0.007165 and $0.007152, output tokens 11,494 and 10,823, the 22.8 second outlier, "output tokens are about 79% of the cost", and "at most about 19%". One-off command. I re-ran the 79% and 19% figures for this file (79.1% and 18.8%).
12. results.md, Phase 5: "There were no SQL repairs in the 50 requests run after repair logging was added" (one-off check).
13. results.md, "Dev fixes and eval v2", and README "Limits": "about 440 successful calls in the quota day" and "`gemini-3.6-flash` allowed 17". This is a one-off count over `logs/llm_calls.jsonl` with an assumed quota-day boundary. The API's own error message says the limit is 500 per day (section 12). The 440 figure should not be used.
14. results.md, Phase 6: the price check against the raw pricing page, the clean-checkout test, the real 502 and 503 error paths, the memory figures (220, 624 and 71 MB), "about 35 seconds" to import sentence-transformers, and the browser check of the UI. All were one-off scripts or manual checks. The parquet rebuild claim (36 of 36 gold queries) is covered by `tests/test_app_and_deploy_data.py`.
15. results.md, Phase 5 and README: "the caching docs list a minimum of 4,096 tokens for implicit caching on gemini-3.6-flash". This comes from reading the Gemini docs through a page summariser, not from a command.

**Stale or wrong statements.**

16. results.md, the "Which model produced which result" section: "Phase 4 will rerun every component on one model". Stale, because Phase 4 did this.
17. results.md, Phase 1: "retail is assumed when the customer type is not stated, and the answer says so once". Stale, because that prompt rule was replaced after eval v1.
18. results.md, Phase 5, "Price table" bullet: "I read the prices ... through a page summariser, so please check them". Superseded, because Phase 6 checked them against the raw page.
19. results.md, "Model, quota and latency": "Its daily limit is not shown to me". Now known: the API error says 500 per day.
20. results.md, "Dev fixes and eval v2": "Those 10 say nothing about the fixes". Partly wrong. The router prompt changed, and those 10 data questions were all routed correctly after the change. Also, one of them (q10) ended with a garbled error message as its answer (section 12).
21. results.md, section "Evaluation dev_after_fixes": the line "dev 10 ({'data': 15, 'policy': 20, ...})" shows the planned counts next to the 10 questions that finished, which is confusing. Its "Answers the verifier changed 1/10" was caused by an error message, not by a model mistake (section 12).
22. README, "Eval v2 (pending)": says the v2 set was committed "after the router and prompt fixes were made". **Wrong.** It was committed before the fixes (commit `da8dea5` comes before the fix commit `83cea5a`). results.md has the correct order.
23. README, "Results": "Every number below is copied from docs/results.md by scripts/sync_readme.py". Only the tables are copied by the script. Prose numbers in the README (79%, 19%, about 440 calls, about 700 MB, about 130 MB, 10 to 25 seconds, about 85%) were typed by hand. I checked two of them for this file: the embedding model is 129 MB on disk (`du`), and 79% matches the re-run (79.1%). The "about 440 calls" figure is wrong (item 13).
24. README, "Limits": "Live demo answers take roughly 10 to 25 seconds". This comes from eval wall times, which include 4.5 second pacing gaps. It was not measured in a deployed demo, because there is none.
