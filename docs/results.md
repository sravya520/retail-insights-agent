# Results

Every number below was produced by the command shown next to it. Sections between the BEGIN and END markers are rewritten by the scripts.

## Which model produced which result

| Result | Model | Notes |
|---|---|---|
| Retrieval comparison (Phase 1) | No Gemini model. Embeddings `BAAI/bge-small-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` | Runs locally. |
| RAG example answers (Phase 1) | `gemini-3.5-flash-lite`, RAG thinking `medium` | Run before the cache existed. |
| SQL accuracy (Phase 2) | `gemini-3.5-flash-lite`, SQL thinking `low` | Cache off. |
| Phase 4 evaluation (all components, run `run-phase4`) | `gemini-3.5-flash-lite` for router, SQL agent, RAG agent and combiner. Judge: `gemini-3.6-flash` for 15 held-out answers, `gemini-3.5-flash-lite` for the other 37 | Cache off. One model for every answering component. |
| Phase 5 caching experiment and token report | `gemini-3.5-flash-lite` for every component | Cache off. 20 held-out v1 questions, no judge. |
| Graph example runs (Phase 3) | `gemini-3.5-flash-lite` for router (low), SQL agent (low), RAG agent (medium), combiner (medium) | Cache on, so some calls in the examples came from cache. |

Nothing in this file was produced by `gemini-3.8-flash`. Its calls early in Phase 1 were discarded. Phase 4 will rerun every component on one model, so the numbers can be compared.

## Phase 1: Documents and RAG

What was built:

- 6 synthetic policy documents for the fictional Northwind Gifts, as Markdown and PDF (2 to 3 pages each), 182 clauses in total.
- Ingestion with `pypdf`, chunked by numbered clause, with `doc`, `section`, `clause_id` and `page` kept. Embeddings with `BAAI/bge-small-en-v1.5` in Chroma.
- Hybrid retrieval (BM25 plus vector, RRF k=60) with an optional cross-encoder reranker.
- A RAG agent with a strict citation format and a citation parser that checks each citation against the retrieved chunks.
- Tests: see the Phase 2 section for the current `pytest` count.

What did not work or is not done:

- **Example answers were generated in a second run, after two fixes.** The first run (reranker on, original prompt) said "The policy documents do not cover this" for the Spain shipping question. Cause: the reranker pushed Shipping §3.2 out of the top 5, and the model did not link Spain to "EU". Fixes: the examples now use plain hybrid search (the best method in the eval), and the prompt lets the model map a country to a region named in the policy. The first run is not kept. I did not fix the model changing its answer on a rerun, so the examples show one run only.
- The configured model `gemini-2.5-flash` returned a 404 (no longer available). See "Model, quota and latency" below for what changed after that.
- The mug example used to add a wholesale case the question did not ask for. The prompt now says to answer only the case asked (retail is assumed when the customer type is not stated, and the answer says so once). The current examples are from a run with the new prompt.
- Question p10 ("My item stopped working after 3 months...") is missed by every method. The gold clause is Warranty 5.2.
- The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) did not improve recall and lowered MRR@5 from 0.850 to 0.771 on this set. It already reranked the top 20 hybrid candidates and kept the top 5 in the Phase 1 run. The pool size is now an explicit setting (`RERANK_POOL = 20`). Re-running with it gives the same numbers, and a pool of 40 is slightly lower (see the table). With only 20 questions, one question moving one rank changes MRR by a visible amount, so treat small differences as noise.
- The gold labels were written by hand before the run. They were not tuned after seeing results.
- Setup note: `pip` and the Hugging Face download failed with `CERTIFICATE_VERIFY_FAILED` because the antivirus on this machine inspects HTTPS. A CA bundle with the antivirus certificate fixed it. This is a local setup issue, not a code issue.

MRR@5 means the reciprocal rank of the first gold clause within the top 5, and 0 if it is not there. A question can list more than one acceptable gold clause.

<!-- BEGIN phase1_retrieval -->
### Retrieval comparison (20 policy questions)

Command: `python evals/retrieval_eval.py`

| Method | Recall@5 | MRR@5 |
|---|---|---|
| BM25 only | 0.70 | 0.608 |
| Vector only | 0.95 | 0.800 |
| Hybrid (RRF k=60) | 0.95 | 0.850 |
| Hybrid + reranker (Phase 1 row) | 0.95 | 0.771 |
| Hybrid + reranker, pool 20 -> top 5 (explicit) | 0.95 | 0.771 |
| Hybrid + reranker, pool 40 -> top 5 | 0.95 | 0.762 |

Questions with no gold clause in the top 5:

- BM25 only: p03, p05, p06, p10, p14, p19
- Vector only: p10
- Hybrid (RRF k=60): p10
- Hybrid + reranker (Phase 1 row): p10
- Hybrid + reranker, pool 20 -> top 5 (explicit): p10
- Hybrid + reranker, pool 40 -> top 5: p10
<!-- END phase1_retrieval -->

<!-- BEGIN phase1_examples -->
### Example answers

Command: `python evals/rag_examples.py`

**Q:** Can a wholesale customer return a damaged item after 40 days?

**A:** No, wholesale customers must report damage within 48 hours of delivery, and claims made after 48 hours may be refused [Returns and Refunds §7.4, p.2].

Citations matched to retrieved chunks: 1. Not matched: 0.

---

**Q:** I bought a personalised mug on 5 December. Can I return it in January because I changed my mind?

**A:** For a retail customer, personalised items cannot be returned for change of mind [Returns and Refunds §3.1, p.1]. Although items delivered between 1 November and 24 December have an extended Christmas return window until 31 January [Returns and Refunds §2.3, p.1], personalised items are excluded from change of mind returns [Returns and Refunds §3.1, p.1].

Citations matched to retrieved chunks: 2. Not matched: 0.

---

**Q:** How much does it cost to ship a £150 order to Spain, and who pays customs charges?

**A:** For a retail customer, shipping a £150 order to Spain is free because EU standard delivery is free on orders of £120 or more [Shipping and Delivery §3.1, p.1]. The customer is responsible for any import duties, taxes, or customs fees charged by the destination country, and Northwind Gifts does not pay these charges [Shipping and Delivery §3.2, p.1].

Citations matched to retrieved chunks: 2. Not matched: 0.

---

**Q:** Do you offer a price match against other shops?

**A:** The policy documents do not cover this.

Citations matched to retrieved chunks: 0. Not matched: 0.
<!-- END phase1_examples -->

<!-- BEGIN phase2_sql -->
### SQL agent execution accuracy (15 questions, all dev split)

Command: `python evals/sql_eval.py`. Model: `gemini-3.5-flash-lite`, thinking level for the SQL agent: `low`.

- **Execution accuracy: 14/15 = 93.3%** (agent result matches gold result, numeric tolerance 0.5%).
- Correct on the first query: 14. Correct after the one repair: 0. Queries that failed even after repair: 0.
- Time per question (writing, running and explaining): median 15.0s, max 22.8s.

| ID | Question | Result | Repaired | Seconds |
|---|---|---|---|---|
| q01 | What was the total revenue in 2010? | pass | no | 15.0 |
| q02 | What were the top 5 products by revenue in November 2011? | pass | no | 11.5 |
| q03 | How many different customers placed an order in 2010? | pass | no | 14.3 |
| q04 | Which country outside the United Kingdom had the highest revenue, and how much? | pass | no | 17.4 |
| q05 | How many orders were placed in March 2011? | pass | no | 11.6 |
| q06 | What was the average order value in 2011? | pass | no | 15.8 |
| q07 | Which month of 2010 had the highest revenue? | pass | no | 17.2 |
| q08 | How many orders were returned in December 2010? | pass | no | 13.2 |
| q09 | What was the total value of returns in 2011? | pass | no | 15.1 |
| q10 | Which customer spent the most in 2010, and how much did they spend? | pass | no | 16.9 |
| q11 | What percentage of 2010 revenue came from orders with no customer ID? | pass | no | 16.3 |
| q12 | Which product sold the most units in 2010? | pass | no | 22.8 |
| q13 | How many units of the WHITE HANGING HEART T-LIGHT HOLDER were sold in 2011? | pass | no | 10.1 |
| q14 | Which 3 customers placed the most orders in 2011? | pass | no | 7.4 |
| q15 | What was the revenue from Germany in each quarter of 2010? | FAIL | no | 13.6 |

Failures in detail:

**q15** What was the revenue from Germany in each quarter of 2010?
- Gold SQL: `SELECT (invoice_month - 1) // 3 + 1 AS quarter, SUM(revenue) AS revenue FROM sales WHERE country = 'Germany' AND invoice_year = 2010 GROUP BY quarter ORDER BY quarter`
- Agent SQL: `SELECT CASE WHEN invoice_month BETWEEN 1 AND 3 THEN 'Q1' WHEN invoice_month BETWEEN 4 AND 6 THEN 'Q2' WHEN invoice_month BETWEEN 7 AND 9 THEN 'Q3' WHEN invoice_month BETWEEN 10 AND 12 THEN 'Q4' END AS quarter, SUM(revenue) AS revenue FROM sales WHERE country = 'Germany' AND invoice_year = 2010 GROUP BY quarter ORDER BY quarter LIMIT 200`
- Gold rows: [[1, 38384.44000000005], [2, 48470.220999999954], [3, 47957.59000000007], [4, 72727.36000000006]]
- Agent rows: [['Q1', 38384.44000000005], ['Q2', 48470.220999999954], ['Q3', 47957.59000000007], ['Q4', 72727.36000000006]]
- Error: None

### SQL example triples

**Question:** What were the top 5 products by revenue in November 2011?

```sql
SELECT p.description, SUM(s.revenue) AS total_revenue FROM sales AS s JOIN products AS p ON s.stock_code = p.stock_code WHERE s.invoice_year = 2011 AND s.invoice_month = 11 AND s.is_merchandise GROUP BY p.description ORDER BY total_revenue DESC LIMIT 5
```

**Answer:** In November 2011, the top product by revenue was RABBIT NIGHT LIGHT with £34,556.72, followed by PAPER CHAIN KIT 50'S CHRISTMAS with £28,985.04. 
The next highest earners were WHITE HANGING HEART T-LIGHT HOLDER at £14,362.12, SMALL POPCORN HOLDER at £14,195.60, and REGENCY CAKESTAND 3 TIER at £13,799.58.

Result matched gold: yes

**Question:** How many orders were returned in December 2010?

```sql
SELECT COUNT(DISTINCT invoice_no) AS orders_returned FROM returns WHERE invoice_year = 2010 AND invoice_month = 12 LIMIT 200
```

**Answer:** There were 326 orders returned in December 2010.

Result matched gold: yes

**Question:** What was the revenue from Germany in each quarter of 2010?

```sql
SELECT CASE WHEN invoice_month BETWEEN 1 AND 3 THEN 'Q1' WHEN invoice_month BETWEEN 4 AND 6 THEN 'Q2' WHEN invoice_month BETWEEN 7 AND 9 THEN 'Q3' WHEN invoice_month BETWEEN 10 AND 12 THEN 'Q4' END AS quarter, SUM(revenue) AS revenue FROM sales WHERE country = 'Germany' AND invoice_year = 2010 GROUP BY quarter ORDER BY quarter LIMIT 200
```

**Answer:** In 2010, Germany generated £38,384.44 in Q1, £48,470.22 in Q2, £47,957.59 in Q3, and £72,727.36 in Q4.

Result matched gold: NO
<!-- END phase2_sql -->

<!-- BEGIN llm_call_latency -->
### Per-call LLM latency and tokens (from `logs/llm_calls.jsonl`)

Command: `python evals/call_latency.py`. Latency is the time inside the model call only. Pacing gaps and retry waits are not counted. Older log lines (before this was split out) include pacing, so their latency is higher than the true call time.

| Model | Node | Calls | Failed | Median s | p95 s | Avg input tokens | Avg output tokens | Avg reasoning tokens |
|---|---|---|---|---|---|---|---|---|
| gemini-3.5-flash-lite | rag_agent | 4 | 0 | 7.9 | 10.5 | 553 | 815 | 760 |
| gemini-3.5-flash-lite | sql_agent.explain | 15 | 0 | 6.5 | 14.2 | 165 | 38 | 0 |
| gemini-3.5-flash-lite | sql_agent.write | 15 | 0 | 7.7 | 11.5 | 896 | 51 | 0 |
<!-- END llm_call_latency -->


## Model, quota and latency (read this before trusting any timing)

- The default model is now `gemini-3.5-flash-lite` (set in `src/ria/config.py`). `gemini-3.8-flash` was tried first. Its free tier allows only 20 requests per day, which I used up, and single calls took 224 to 327 seconds in one run. `gemini-3.5-flash-lite` allows 15 requests per minute on the free tier (the API error said so). Its daily limit is not shown to me.
- Thinking level (from the Gemini docs, setting `thinking_level`): router `low`, SQL agent `low`, RAG agent `medium`. At the default (minimal) level the lighter model wrongly answered "not covered" for the wholesale damage question, although retrieval had the right clause first. This was seen once on one question and not tested further.
- Call latency swings a lot with API load. The first SQL eval run took about 2 seconds per question. The final run took a median of 15 seconds per question (7 to 8 seconds per call). I did not control for this, so treat all latency numbers as one sample.
- Tenacity now waits longer between retries (random up to 4, 8, 16 and 32 seconds) and a 4.5 second gap between calls keeps the free tier under 15 requests per minute. A 429 that says the daily quota is used up is not retried, because waiting seconds cannot fix it. This is a small change from "retry on every 429".
- Every LLM call is logged to `logs/llm_calls.jsonl` with node, model, tokens, latency (model time only) and wait time. The log before the fix is kept as `logs/llm_calls_old_pre_fix.jsonl`.

## Phase 2: Sales database and SQL agent

What was built (all run by the commands shown):

- Dataset: UCI Online Retail II, downloaded from the UCI site into `data/raw/` (git-ignored). Command: `python -I -m ria.sql_db`. The load report is in `data/db/load_report.json`.
- Load counts: 1,067,371 raw rows. 19,494 cancellation rows (invoice starts with "C"). 1,041,670 clean sales rows and 19,493 clean returns rows. 6,207 non-cancelled rows were dropped for price or quantity of zero or less. 236,121 sales rows have no customer ID. They are kept, so revenue counts them. Total sales revenue is £20,972,594.57.
- Cleaning rules: `sales` has quantity > 0 and price > 0. `returns` has the cancelled lines with positive units and value. `is_merchandise` separates real products from postage, fees and adjustments. Exact duplicate rows are not removed.
- `sql_agent.py`: the LLM writes DuckDB SQL from the schema text and 3 few-shot examples. Code checks it with `sqlglot` (one SELECT only, known tables and columns only, file and system functions blocked), adds or lowers LIMIT to 200, runs it read-only with file access off and a 5 second timeout, and gives one repair attempt. It returns the SQL, the rows and a plain-English answer.
- Tests: `pytest` gives 56 passed (command: `pytest`). They cover rejecting DROP, DELETE, multiple statements, unknown tables and columns, file reading functions, adding LIMIT, the timeout, the read-only connection, the cleaning rules, the repair loop on a scripted fake LLM, and the result comparison.

Honest notes on the SQL numbers:

- All 15 SQL questions are in the dev split. I wrote them and the gold SQL, and I ran them to check the agent. Held-out questions come in Phase 4 (see `evals/DEV_SPLIT.md`).
- The score of 14/15 is strict. The one miss (q15) is a label difference: the agent returned the quarter as `'Q1'` to `'Q4'` and the gold has `1` to `4`. The revenue numbers are identical. I did not loosen the comparison after seeing this. With a lenient label check the score would be 15/15.
- Gold answers are produced by running my gold SQL on the database (`python evals/build_sql_gold.py`). They are as correct as my SQL is.
- The first full run hit the 15 requests per minute limit on the last two questions (13 of 13 answered passed). That is why the pacing gap exists. The reported run is a complete re-run.
- The question "Which product sold the most units" and the product questions rely on `is_merchandise` and the `products` view. The schema text tells the model so.

<!-- BEGIN phase3_examples -->
### Graph example runs (LangGraph, memory, verifier)

Command: `python evals/graph_examples.py`. Model: `gemini-3.5-flash-lite` for the router, SQL agent, RAG agent and combiner
(thinking: router low, SQL low, RAG and combiner medium). LLM cache: on.
Calls marked "from cache" were answered by the local development cache from an earlier identical call, so the wall time of those runs is not a latency measurement.

**data (turn 1 of the memory example)**

- Thread: `demo-memory`
- Question: What were the top 5 products by revenue in November 2011?
- Standalone question from the router: What were the top 5 products by revenue in November 2011?
- Route: `data`
- Trajectory: `['router', 'sql_agent', 'combiner', 'verifier']`
- Verifier: verified = True, sentences removed = 0
- SQL: `SELECT p.description, SUM(s.revenue) AS total_revenue FROM sales AS s JOIN products AS p ON s.stock_code = p.stock_code WHERE s.invoice_year = 2011 AND s.invoice_month = 11 AND s.is_merchandise GROUP BY p.description ORDER BY total_revenue DESC LIMIT 5`
- Model calls: 1 real, 2 from cache. Wall time 2.145 s.

Answer:

> In November 2011, the top product by revenue was RABBIT NIGHT LIGHT with £34,556.72, followed by PAPER CHAIN KIT 50'S CHRISTMAS with £28,985.04.
> The next highest earners were WHITE HANGING HEART T-LIGHT HOLDER at £14,362.12, SMALL POPCORN HOLDER at £14,195.60, and REGENCY CAKESTAND 3 TIER at £13,799.58.

---

**data follow-up (turn 2, same thread_id)**

- Thread: `demo-memory`
- Question: And in December?
- Standalone question from the router: What were the top 5 products by revenue in December 2011?
- Route: `data`
- Trajectory: `['router', 'sql_agent', 'combiner', 'verifier']`
- Verifier: verified = True, sentences removed = 0
- SQL: `SELECT p.description, SUM(s.revenue) AS total_revenue FROM sales AS s JOIN products AS p ON s.stock_code = p.stock_code WHERE s.invoice_year = 2011 AND s.invoice_month = 12 AND s.is_merchandise GROUP BY p.description ORDER BY total_revenue DESC LIMIT 5`
- Model calls: 1 real, 2 from cache. Wall time 3.772 s.

Answer:

> In December 2011, the top product by revenue was PAPER CRAFT , LITTLE BIRDIE with £168,469.60.
> The next highest-selling products were RABBIT NIGHT LIGHT at £9,643.02, PAPER CHAIN KIT 50'S CHRISTMAS at £6,870.71, REGENCY CAKESTAND 3 TIER at £6,006.87, and BLACK RECORD COVER FRAME at £5,582.43.

---

**policy (dev question p05)**

- Thread: `demo-policy`
- Question: Can a wholesale customer return a damaged item after 40 days?
- Standalone question from the router: Can a wholesale customer return a damaged item after 40 days?
- Route: `policy`
- Trajectory: `['router', 'rag_agent', 'combiner', 'verifier']`
- Verifier: verified = True, sentences removed = 0
- Model calls: 1 real, 1 from cache. Wall time 17.387 s.

Answer:

> No, a wholesale customer cannot return a damaged item after 40 days because damage must be reported within 48 hours of delivery [Returns and Refunds §7.4, p.2]. Claims made after 48 hours may be refused [Returns and Refunds §7.4, p.2], and the wholesale return window is 14 days [Returns and Refunds §7.1, p.2].

---

**both**

- Thread: `demo-both`
- Question: How many orders were returned in December 2010, and how long do retail customers have to return an item if they change their mind?
- Standalone question from the router: How many orders were returned in December 2010, and how long do retail customers have to return an item if they change their mind?
- Route: `both`
- Trajectory: `['router', 'sql_agent', 'rag_agent', 'combiner', 'verifier']`
- Verifier: verified = True, sentences removed = 0
- SQL: `SELECT COUNT(DISTINCT invoice_no) AS returned_orders FROM returns WHERE invoice_year = 2010 AND invoice_month = 12 LIMIT 200`
- Model calls: 5 real, 0 from cache. Wall time 20.986 s.

Answer:

> There were 326 returned orders in December 2010, and retail customers have 30 days from the delivery date to return an unused item if they change their mind [Returns and Refunds §2.1, p.1].

---

**out_of_scope**

- Thread: `demo-oos`
- Question: What is the weather in London tomorrow?
- Standalone question from the router: What is the weather in London tomorrow?
- Route: `out_of_scope`
- Trajectory: `['router', 'refuse']`
- Verifier: verified = None, sentences removed = 0
- Model calls: 1 real, 0 from cache. Wall time 2.839 s.

Answer:

> I can only help with questions about Northwind Gifts sales data and its policy documents (returns, warranty, shipping, wholesale terms, loyalty programme, data privacy). I cannot help with that question.

---

**Verifier test with an injected fault (not model output)**

I took the real draft answer of the `both` run and appended two made-up sentences: one with a number that is in no data or policy text (87%), and one with a citation that matches no retrieved chunk (§9.9).

- Removed: [{"sentence": "Revenue also grew 87% on last year.", "reason": "unsupported_number", "detail": "87"}, {"sentence": "Wholesale buyers can return goods for 90 days [Returns and Refunds \u00a79.9, p.3].", "reason": "invalid_citation", "detail": "Returns and Refunds \u00a79.9"}]
- verified = False

Answer after verification:

> There were 326 returned orders in December 2010, and retail customers have 30 days from the delivery date to return an unused item if they change their mind [Returns and Refunds §2.1, p.1].

The cleaned answer equals the original draft: True.
<!-- END phase3_examples -->


## Phase 3: LangGraph workflow with memory

What was built (commands shown are the ones that produced the numbers):

- `graph.py`: a LangGraph `StateGraph` with the nodes `router`, `sql_agent`, `rag_agent`, `combiner`, `verifier` and `refuse`. The router uses structured output with a Pydantic class (`RouterDecision`). State is saved per `thread_id` with `MemorySaver`. Each node adds its name to `state["trajectory"]`, and each request writes a line to `logs/traces.jsonl`.
- `verifier.py`: plain code, no LLM. It splits the answer into sentences and removes a sentence if it has a citation that matches no retrieved chunk (doc and clause id), or a number that is not supported. A number is supported if it is in the SQL rows (with rounding tolerance), in the user's question, in the text of a chunk that the same sentence cites, or equals the row count. Removed sentences are returned with the reason and are written to the trace log.
- `llm.py`: a local response cache (flag `LLM_CACHE_ENABLED` in `config.py`, or `RIA_LLM_CACHE=0`). The key is model, full prompt and settings. Cache hits are logged in `logs/llm_calls.jsonl` with `cache_hit: true`. The default is on. `evals/sql_eval.py` turns it off. The final eval and latency runs must turn it off too.
- Tests: `pytest` gives 86 passed (command: `pytest`). New tests cover the verifier (fabricated number, fake citation, rounding, citation after the full stop, list markers, idempotence), the router schema, all four routes end to end on a mocked LLM, memory across turns and across threads, the split of a `both` question, the trace line, and the cache (hit, off switch, key changes, structured calls).

Real API use in Phase 3: 23 model calls in total (10 router, 8 SQL, 3 RAG, 2 combiner), no errors. This is more than the 14 I planned, because I ran the examples three times: one run crashed on a bug, and one run exposed the two problems below. The repeats were partly answered from cache.

What the examples showed (section above, produced by `python evals/graph_examples.py`):

- **All four routes work.** Trajectories were exactly `[router, sql_agent, combiner, verifier]` for data, `[router, rag_agent, combiner, verifier]` for policy, `[router, sql_agent, rag_agent, combiner, verifier]` for both, and `[router, refuse]` for out of scope.
- **Memory works.** On thread `demo-memory`, "And in December?" was rewritten by the router to "What were the top 5 products by revenue in December 2011?". The answer for December 2011 is dominated by one product with £168,469.60. That is real data in this dataset (one very large order), not a bug.
- **The verifier removed nothing in the real runs.** The model did not produce an unsupported number or citation in these 5 runs. To show the removal path I added a test with an injected fault. I appended a made-up number (87%) and a fake citation (§9.9) to a real answer. The verifier removed both sentences, logged why, and the cleaned answer equals the original. This shows the verifier works. It does not show how often the model makes such mistakes. Phase 4 measures that on more questions.

Two real problems found and fixed during this phase:

1. **The SQL agent invented a number.** In the first `both` run the SQL agent wrote `SELECT COUNT(...), 28 AS return_window_days ...`. The 28 came from nowhere (the policy says 30), and the verifier accepted it because it was in the SQL result. Fixes: (a) a new SQL guardrail rejects constant values in the top-level SELECT list, so the agent must repair the query, and (b) the router now splits a `both` question into `data_question` and `policy_question`, so each agent only sees its own part. The fix is unit-tested. I did not measure how often the model invented constants before the fix.
2. **The mixed question confused the RAG agent.** It saw the whole mixed question and answered "The policy documents do not cover this", and the combiner kept that next to the invented 28. After the split, the answer is correct: 326 returned orders, and 30 days with the citation `[Returns and Refunds §2.1, p.1]`.

Also fixed: on a cache miss, Gemini can return reply content as a list of parts, which crashed the combiner. `invoke_with_retry` now always returns text.

Known limits of the verifier (not fixed):

- It trusts the SQL rows. If the SQL is wrong but runs, the number passes. That is what SQL execution accuracy measures.
- It checks numbers and citation ids only. It does not check that a cited chunk supports what the sentence says. Numbers written as words ("five") are not checked.
- It extends the original rule in two ways: policy numbers are allowed if they appear in the cited chunk, and numbers repeated from the question are allowed. Without these, almost every policy answer would be removed.
- `verified` is `True` when nothing had to be removed. If every sentence is removed the answer becomes a fixed fallback message. For out of scope questions `verified` is `None` because no answer text is checked.
- The router and the whole graph are only tested on 5 real questions so far. There are no router or route-accuracy numbers until Phase 4.
- The Phase 2 SQL accuracy (14/15) was measured before the Phase 3 SQL guardrail that rejects constants in the SELECT list and before the thinking and cache changes. I did not re-run it, to save API calls. Phase 4 reruns it.

## Eval v1 (FROZEN, git tag `eval-v1`)

These are the Phase 4 results for the system as of tag `eval-v1` (commit `f3b43ee`). **Do not edit or regenerate them.**
Later changes to the system are evaluated on a new held-out set (v2) and reported in their own section. The code refuses to overwrite this section.

<!-- BEGIN eval_v1 -->
### Phase 4 evaluation (run `run-phase4`)

Status: **COMPLETE**. Command: `python evals/run_evals.py`. Held-out questions were committed before any run in commit `fc2973aff48267cec53ede2965e2f8e6c8f708f0`.

- Answering model for every component: `gemini-3.5-flash-lite` (router low, SQL low, RAG medium, combiner medium thinking). LLM cache: **off**.
- Judge: `gemini-3.5-flash-lite` scored 37, `gemini-3.6-flash` scored 15. By split: heldout: 1 by `gemini-3.5-flash-lite`, 15 by `gemini-3.6-flash`; dev: 36 by `gemini-3.5-flash-lite`, 0 by `gemini-3.6-flash`. Judge models that ran out of free daily quota during the run: `gemini-3.6-flash`, `gemini-3.7-flash`. Some items were self-judged (the answering model scored its own answers), which can inflate scores. See the limits below.
- Real API calls in this run: **206** (152 answering, 54 judge). By node: {'combiner': 7, 'judge': 54, 'rag_agent': 31, 'router': 57, 'sql_agent.explain': 28, 'sql_agent.repair': 1, 'sql_agent.write': 28}. Failed calls: 2.
- Questions: heldout 20 ({'data': 6, 'policy': 6, 'both': 4, 'out_of_scope': 4}), dev 37 ({'data': 15, 'policy': 20, 'both': 1, 'out_of_scope': 1}).

**Held-out is the headline. Dev numbers are optimistic because the system was tuned on those questions.**

#### Gates (must pass)

| Gate | Result | Detail |
|---|---|---|
| No unsupported number or citation in a final answer (verifier re-run on every answer) | PASS | heldout 16/16, dev 36/36 |
| All out of scope questions refused | PASS | heldout 4/4, dev 1/1 |
| SQL guardrail blocks every unsafe query | PASS | 51/51 unsafe queries blocked (`evals/unsafe_sql.json`, no model involved) |

Gate 1 uses the verifier's own rules, so it shows the verifier output is stable. It does not prove an answer is true.
Violating ids: heldout none, dev none. Gate 2 misses: heldout none, dev none.

#### Observations

| Metric | Held-out | Dev |
|---|---|---|
| Router accuracy (route matches expected) | 20/20 = 100% (95% CI low 84%) | 35/37 = 95% |
| Trajectory accuracy (exact node sequence) | 20/20 = 100% (95% CI low 84%) | 35/37 = 95% |
| SQL execution accuracy (data and both questions, 0.5% tolerance) | 10/10 = 100% (95% CI low 72%) | 15/16 = 94% |
| Retrieval recall@5 (policy and both questions, end to end) | 10/10 = 100% (95% CI low 72%) | 20/21 = 95% |
| Refusal rate on out of scope questions | 4/4 = 100% (95% CI low 51%) | 1/1 = 100% |
| Judge faithfulness, mean 1 to 5 | 4.94 (n=16) | 4.92 (n=36) |
| Judge answer relevance, mean 1 to 5 | 5.00 (n=16) | 5.00 (n=36) |
| Judge says faithful (faithfulness 4 or 5) | 100% | 100% |
| Answers the verifier changed | 0/16 | 0/36 |
| Sentences removed (number / citation) | 0 / 0 | 0 / 0 |
| Median wall time per question (s, includes pacing gaps) | 10.4 | 11.2 |

Most common wrong paths (both splits):

- 2x expected router > rag_agent > combiner > verifier | got router > sql_agent > rag_agent > combiner > verifier

Gold queries wrongly blocked by the SQL guardrail: 0 of 26.

Judge calibration (agreement with human labels): **not computed yet.** It needs `evals/human_labels.jsonl` to be filled in by hand.
<!-- END eval_v1 -->


## Phase 4: reading the evaluation above

What was built (all run by `python evals/run_evals.py`, with `--report` to rebuild the tables from saved records):

- `evals/questions.jsonl`: 57 questions, each with expected route, expected trajectory and gold answer (data) or gold citations (policy). **Held-out: 20 questions** (6 data, 6 policy, 4 both, 4 out of scope). **Dev: 37 questions** (15 data, 20 policy, 1 both, 1 out of scope). The dev both and out of scope sets have only one question each because I only had one of each from earlier phases, so dev numbers for those routes mean nothing.
- The held-out questions and the dev set were committed in `fc2973aff48267cec53ede2965e2f8e6c8f708f0` before any eval run. I wrote the held-out questions from the policy documents and the database schema only. No model output was seen. Gold answers for data questions come from running my gold SQL on the database.
- `src/ria/eval_metrics.py`: router, trajectory, SQL, retrieval, refusal and verifier metrics, the three gates, and the human-label and agreement logic. `src/ria/judge.py`: the judge with a fixed 1 to 5 rubric. `evals/unsafe_sql.json`: 51 unsafe queries for the guardrail gate.
- Records of every question are saved in `evals/runs/run-phase4/records.jsonl`. The run stops cleanly and can resume if a daily quota runs out. It did not run out for the answering model.
- Tests: `pytest` gives 101 passed (command: `pytest`).

Real API calls: **206** in the run (152 answering, 54 judge). Two of the 54 judge calls failed on the daily quota of the first two judge models. A smoke test with 2 questions before the run used 8 more calls and is not in the 206.

### Gates

All three gates passed. Read gate 1 with care. The verifier changed **0 of 52** real answers, so there was nothing for it to remove. Gate 1 therefore shows that the final answers are stable under the verifier's rules, and not that the verifier catches mistakes in practice. Its removal behaviour is shown only by unit tests and by the injected-fault test in Phase 3. This run gives no real example of the model producing an unsupported number or citation after the Phase 3 fixes. The human-label sample therefore has no verifier-changed answers.

### SQL guardrail check after the constant rule

- The constant guardrail blocked **0 of the 26** gold queries (15 dev, 11 held-out gold queries including the data part of the both questions). The q15 labels `'Q1'` to `'Q4'` were never blocked, because they sit inside a `CASE` and are not a bare constant.
- The rule did block a plain string label such as `SELECT 'Germany' AS country, SUM(revenue) ...`, which is legitimate. As you asked, I narrowed the rule to numeric constants in the top-level SELECT list. Tests cover both cases: string labels and CASE labels are allowed, and `28 AS window_days` and `SELECT 42` are rejected.
- Scores: before the guardrail (Phase 2, 15 dev data questions) 14/15. After the guardrail and the narrowing (this run, same 15 questions) 14/15. The same question fails both times: q15, where the agent returns the labels `'Q1'` to `'Q4'` and the gold has `1` to `4`. The revenue numbers are identical. I kept the comparison strict.
- One query in the run (held-out h04) needed the one repair attempt and then passed. I did not record why the first query was rejected, so I cannot say whether the guardrail caused it. I should log the repair reason in the next phase.

### What the numbers do and do not show

- **Held-out is 100% on every metric, but the sets are small.** The 95% lower bounds are 84% for 20 questions and 72% for 10. A perfect score on 20 questions is consistent with a true rate of about 85%. My held-out questions may also be easier than real user questions, because they are clean, single-topic and in plain English.
- **Two real router errors, both on dev questions.** p07 ("Does the extended Christmas returns period also apply to business accounts?") and p19 ("What happens to my loyalty points if I return something I bought?") were routed to `both` instead of `policy`. The system then ran an unneeded SQL query. The p07 answer starts with an irrelevant sentence about return counts by country. This is the only wrong trajectory in the run.
- **Retrieval:** p10 still misses its gold clause (Warranty 5.2), as in Phase 1. Every other policy and both question retrieved a gold clause in its top 5.
- **The judge is lenient and mostly self-judging.** The judge gave answer relevance 5 to every answer, including p07 with its irrelevant sentence, so the relevance mean of 5.00 is not credible. For the held-out split, 15 of 16 answers were judged by `gemini-3.6-flash` and 1 by the answering model. For the dev split, all 36 were judged by the answering model, because `gemini-3.6-flash` and `gemini-3.7-flash` ran out of free daily quota. Self-judging can inflate scores, so the dev judge scores are weak evidence. Temperature 0 was requested, but `gemini-3.5-flash-lite` uses fixed sampling settings and ignores it, so its scores are not guaranteed to repeat.
- **Calibration is pending.** The judge's agreement with human labels is not computed. See `evals/HUMAN_LABELS.md`.
- **My own prompt adds an unsupported phrase.** All faithfulness scores below 5 come from "For a retail customer, ..." added by the RAG prompt rule to answers where the question did not say that (held-out h11, dev p14 and p15) and from one rounding note (q04). That phrase is not in the cited passage. I did not change the prompt, because h11 is a held-out question and fixing the prompt after seeing it would make held-out no longer held-out.
- The policy `gold_text` field is not used by any metric. Policy answers are scored by gold citation recall and the judge only, so a policy answer with the right citations and a wrong statement would not be caught except by the judge.
- Wall times include the 4.5 second pacing gaps, so they are not model latency. Phase 5 reports per-call latency.

<!-- BEGIN phase5_caching -->
### Caching experiment (20 held-out v1 questions, LLM response cache off)

Command: `python evals/caching_experiment.py`. Model: `gemini-3.5-flash-lite` for every component. Complete.
Real API calls in this experiment: **112** (including the repeat probe).

`question_first` is a comparison layout I built for this test (question first, then instructions, schema and examples).
`stable_first` is the production layout (instructions, schema and examples first, question last). The baseline never ran in production.
The two layouts ran one after the other (question_first first), so API load and time of day can differ between them.

| Layout | Queries | LLM calls | Input tokens | Cached input tokens | Cost per query | Cost per 1,000 queries | Median call latency (s) | Median model time per query (s) |
|---|---|---|---|---|---|---|---|---|
| question_first (before) | 20 | 54 | 23882 | 0 (0.0%) | $0.001795 | $1.795 | 1.38 | 4.00 |
| stable_first (after) | 20 | 54 | 23839 | 0 (0.0%) | $0.001710 | $1.710 | 1.26 | 3.47 |

Change from before to after: cached tokens n/a, cost per query -8.452e-05 (-4.7%), median call latency -0.124 (-9.0%), median model time per query -0.5315 (-13.3%).

Cost uses the price table in `config.PRICES` (paid-tier list prices, thinking tokens billed as output). Input tokens per layout differ only by a few tokens, because the same text is sent in a different order.

Average tokens per call by node and layout:

| Layout | Node | Calls | Avg input | Avg output |
|---|---|---|---|---|
| question_first | combiner | 4 | 168 | 736 |
| question_first | rag_agent | 10 | 530 | 569 |
| question_first | router | 20 | 374 | 91 |
| question_first | sql_agent.explain | 10 | 146 | 65 |
| question_first | sql_agent.write | 10 | 897 | 40 |
| stable_first | combiner | 4 | 159 | 752 |
| stable_first | rag_agent | 10 | 532 | 535 |
| stable_first | router | 20 | 373 | 90 |
| stable_first | sql_agent.explain | 10 | 147 | 27 |
| stable_first | sql_agent.write | 10 | 896 | 40 |

Repeat probe: the identical SQL prompt (894 input tokens) sent 4 times in a row, production layout:

| Call | Input tokens | Cached tokens | Latency (s) |
|---|---|---|---|
| 1 | 894 | 0 | 1.09 |
| 2 | 894 | 0 | 0.78 |
| 3 | 894 | 0 | 1.27 |
| 4 | 894 | 0 | 1.21 |

Sanity check that the layout did not change behaviour (20 questions, not a quality measurement): router correct (20, 20) before and (20, 20) after; exact trajectory (20, 20) before and (20, 20) after; SQL execution accuracy (10, 10) before and (9, 10) after (as correct / total).
<!-- END phase5_caching -->

<!-- BEGIN phase5_explicit_cache -->
### Explicit context caching trial (SQL system prompt)

Command: `python evals/explicit_cache_trial.py`. The cached content is the SQL system prompt (schema, rules and 3 few-shot examples).

| Model | Prompt | Prompt tokens | Result |
|---|---|---|---|
| `gemini-3.5-flash-lite` | SQL system prompt | 876 | cache NOT created: ClientError: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Cached content is too small. total_token_count=879, min_total_token_count=1024', 'status': 'INVALID_ARGUMENT'}} |
| `gemini-3.6-flash` | SQL system prompt | 876 | cache NOT created: ClientError: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': 'Cached content is too small. total_token_count=879, min_total_token_count=1024', 'status': 'INVALID_ARGUMENT'}} |
| `gemini-3.5-flash-lite` | padded prompt (synthetic: schema repeated once) | 1465 | cache NOT created: ClientError: 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'TotalCachedContentStorageTokensPerModelFreeTier limit exceeded for model gemini-3.5-flash-lite: limit=0, requested=1468', 'status': 'RESOURCE_EXHAUSTED'}} |
<!-- END phase5_explicit_cache -->

<!-- BEGIN phase5_tokens -->
### Token economics (per route)

Command: `python evals/token_report.py`. Source records: `caching-stable_first.jsonl` (20 requests: data 6, policy 6, both 4, out_of_scope 4). Model: `gemini-3.5-flash-lite` for every component,
production prompt layout, LLM response cache off. The judge is not included (it is an evaluation cost, not a serving cost).
Cost uses `config.PRICES` (paid-tier list prices from the Gemini pricing page; thinking tokens are billed as output). The free tier bills nothing.
Route groups are small (4 to 12 requests), so treat the averages as rough.

| Route | Requests | LLM calls per query | Avg input tokens | Avg output tokens | Avg total tokens | Cost per query | Cost per 1,000 queries |
|---|---|---|---|---|---|---|---|
| data | 6 | 3.0 | 1421 | 160 | 1581 | $0.000827 | $0.827 |
| policy | 6 | 2.0 | 898 | 645 | 1543 | $0.001881 | $1.881 |
| both | 4 | 5.0 | 2117 | 1432 | 3548 | $0.004214 | $4.214 |
| out_of_scope | 4 | 1.0 | 365 | 66 | 431 | $0.000275 | $0.275 |
| **all (this mix)** | 20 | 2.7 | 1192 | 541 | 1733 | $0.001710 | $1.710 |

Output tokens (which include thinking tokens) are 31% of all tokens in these requests, and cost 8.3 times more per token than input tokens.

Average tokens per call, by node:

| Node | Calls | Avg input | Avg output |
|---|---|---|---|
| combiner | 4 | 159 | 752 |
| rag_agent | 10 | 532 | 535 |
| router | 20 | 373 | 90 |
| sql_agent.explain | 10 | 147 | 27 |
| sql_agent.write | 10 | 896 | 40 |

Share of tokens spent on the schema and on retrieved context. Fixed prompt parts were counted with the Gemini `count_tokens` endpoint
(schema description 580 tokens, few-shot examples 207, whole SQL system prompt 876). The retrieved context is the
policy passages sent to the RAG agent, counted per request. Counts are approximate because parts are counted separately from the whole prompt.
Total input tokens in these requests: 23839. Total tokens including output: 34662.

| Part of the prompt | Tokens | Share of input tokens | Share of all tokens (input + output) |
|---|---|---|---|
| Schema description (sent in every SQL write call) | 5800 | 24.3% | 16.7% |
| Few-shot SQL examples | 2070 | 8.7% | 6.0% |
| Other SQL instructions and rules | 890 | 3.7% | 2.6% |
| Retrieved policy context (5 passages in every RAG call) | 2664 | 11.2% | 7.7% |
<!-- END phase5_tokens -->


## Phase 5: observability and token economics

What was built:

- **Tracing** (`src/ria/tracing.py`). Every LLM call is logged to `logs/llm_calls.jsonl` with node, model, input, output and cached tokens, reasoning tokens, latency (model time only), wait time, cost and cache hit. Every request is logged to `logs/traces.jsonl` with the question, route, trajectory, verified flag, total tokens, total cost, model latency, wall time and the list of its calls. Cache hits add no tokens, cost or latency to a request total. SQL repairs are logged to `logs/sql_repairs.jsonl` with the stage that failed (validation or execution), the failed SQL and the error.
- **Price table** in `config.PRICES` (USD per million tokens, editable): `gemini-3.5-flash-lite` input 0.30, output 2.50, cached input 0.03. I read the prices from the Gemini pricing page through a page summariser, so please check them against the page. The free tier bills nothing, so all costs below are what the same usage would cost on a paid plan.
- **Langfuse**: skipped. No Langfuse keys are in `.env`. The code attaches the Langfuse callback only when both keys are set and the package imports, and never raises. Tests cover the skip and the failure case. I have not tested it against a live Langfuse account.
- **Reports**: `evals/token_report.py` (section "Token economics"), `evals/caching_experiment.py` (section "Caching experiment"), `evals/explicit_cache_trial.py` (section "Explicit context caching trial").
- **Nightly job** `.github/workflows/nightly-eval.yml`: installs, builds the vector index and the DuckDB database, runs the offline tests, then runs a fixed 10-question subset (`evals/nightly_subset.json`: 3 data, 3 policy, 2 both, 2 out of scope, all from the v1 held-out set) on the real model with the cache off, no judge and no write to this file. The job fails if a test fails or a gate fails. I cannot run GitHub Actions from here (this folder has no remote). What I did run: the YAML parses, the exact eval command ran locally on all 10 questions (10 of 10 questions answered, all three gates passed, exit code 0, 30 real API calls), and a test (`tests/test_nightly_gate.py`) proves that an answered out of scope question gives exit code 1. The Hugging Face model download and the dataset download steps in the workflow are untested in CI.
- Eval v1 is tagged `eval-v1` and its results section is frozen. The code refuses to overwrite it.
- Tests: `pytest` gives 118 passed at the end of the core Phase 5 work (see the next section for the current count).

### Reading the caching experiment

- **Correction found after the experiment ran.** The question-first baseline layout was applied to the router and the SQL write prompts only. The RAG prompt was not switched, because a patch of mine silently did not apply, so the RAG calls were identical in both runs (average input 530 and 532 tokens). I fixed this afterwards and did not re-run the experiment, to save quota. The conclusion does not depend on it: no prompt reached a size where caching can trigger, and the RAG prompt is about 530 tokens in either layout.
- **Caching did not trigger on this model.** Cached tokens were 0 in both layouts and 0 in the identical-prompt probe (the same 894-token SQL prompt sent 4 times in a row). The SQL prompt is far smaller than the minimum the caching docs list for implicit caching on the newer Flash models (4,096 tokens on `gemini-3.6-flash`). The docs give no minimum for `gemini-3.5-flash-lite`. My prompts are 150 to 900 tokens.
- **Explicit caching is blocked in two ways.** The API refuses to cache the SQL prompt because it has 879 tokens and the minimum is 1,024. A padded prompt that passes the minimum is refused too, because the free tier has a cached storage limit of 0 for this model. So explicit caching is not possible here without a paid plan, and a longer prompt.
- **The before and after difference is not a caching effect.** Input tokens are almost the same in both layouts (23,882 and 23,839 tokens, input cost $0.007165 and $0.007152). The 4.7% lower cost per query in the second run comes from fewer output tokens (11,494 and 10,823), which are mostly thinking tokens and vary from run to run. The latency difference has the same cause plus one 22.8 second outlier in the first run. I would not claim any improvement from the prompt layout.
- **Even perfect caching would help little.** Output tokens are about 79% of the cost in these runs and input tokens about 21%. Cached input is billed at one tenth of the input price here. So if every input token were cached, the saving would be at most about 19% of cost. Only the repeated static parts of the prompts could be cached, so the real ceiling is lower.
- **The biggest cost lever is thinking tokens, not the prompt.** RAG and combiner calls average 535 and 752 output tokens for answers of a few sentences, because their thinking level is `medium`. I did not change this. Lowering it would change answer quality, which I have not measured.
- **The model is not fully repeatable.** On held-out question h14 ("total value of returns in 2010") the SQL agent returned the gold answer in the first run and in the Phase 4 run, but in the second experiment run it added `is_merchandise = TRUE`, which gave a different number (245,842 instead of 679,179). Temperature 0 is requested but this model ignores it. So single-run accuracy numbers, including the 100% SQL accuracy in eval v1, carry run-to-run noise that I have not measured.

### Reading the token report

- Per query, on this 20 question mix, the cost is about $0.0017 ($1.71 per 1,000 queries). By route, per 1,000 queries: data $0.83, policy $1.88, both $4.21, out of scope $0.28. A `both` question costs about 15 times an out of scope question, because it makes 5 calls and the RAG and combiner calls produce long thinking output.
- The schema description is 24% of all input tokens (17% of all tokens). The retrieved policy context is 11% of input tokens. The few-shot examples are 9%. Output tokens (thinking included) are 31% of all tokens.
- Route groups hold 4 to 6 requests each, so the per-route numbers are rough. The token report is run again at the end of this phase on a larger set (see the v2 section).
- There were no SQL repairs in the 50 requests run after repair logging was added, so `logs/sql_repairs.jsonl` does not exist yet. The reason for the earlier h04 repair is still unknown. The repair logger has been tested only with scripted fake models.

<!-- BEGIN phase5_rewrite_experiment -->
### Query rewrite experiment (dev policy questions p01 to p20, retrieval only)

Command: `python evals/rewrite_experiment.py`. Model for the rewrite: `gemini-3.5-flash-lite` (thinking low). LLM cache on (dev experiment).
Decision rule written before the run: adopt the fused search only if MRR@5 rises by at least 0.03 and recall@5 does not drop.

| Variant | Recall@5 | MRR@5 |
|---|---|---|
| baseline | 0.95 | 0.850 |
| rewrite | 0.90 | 0.739 |
| fused | 0.95 | 0.877 |

**Verdict: DO NOT ADOPT (rule not met).** Each rewrite call averaged 92 input and 7 output tokens ($0.000045, 0.9 s median) over 20 real calls.

Questions whose gold rank changed between baseline and fused (rank 1 is best, None means not in the top 5):

- p01 (baseline 2 -> fused 3): How many days do I have to return an item if I just changed my mind? | rewritten as: return policy timeframe change of mind
- p03 (baseline 1 -> fused 5): My vase broke in the post. How long do I have to report it and what proof do you need? | rewritten as: broken item delivery damage reporting time limit proof photo policy
- p14 (baseline 2 -> fused 1): Who pays import duty on a parcel sent to France? | rewritten as: import duty responsibility France international shipping taxes fees
- p16 (baseline 2 -> fused 1): What is the smallest first order a new trade account can place? | rewritten as: minimum first order trade account wholesale threshold
- p19 (baseline 2 -> fused 1): What happens to my loyalty points if I return something I bought? | rewritten as: loyalty points return refund policy

These are dev questions, so this result is optimistic. Whether it holds on unseen questions is measured on eval v2.
<!-- END phase5_rewrite_experiment -->

<!-- BEGIN dev_after_fixes -->
### Evaluation dev_after_fixes (run `run-dev-after-fixes`)

Status: **PARTIAL: 10 of 37 questions finished (stopped on error: Gemini call failed (GoogleRateLimitError): Error calling model 'gemini-3.5-flash-lite' (RESOURCE_EXHAUSTED): 429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your current quota)**. Command: `python evals/run_evals.py` with questions from `questions.jsonl`. These held-out questions were committed before any run, in commit `fc2973aff48267cec53ede2965e2f8e6c8f708f0`.

- Answering model for every component: `gemini-3.5-flash-lite` (router low, SQL low, RAG medium, combiner medium thinking). LLM cache: **off**.
- Judge: no items were judged. By split: heldout: none; dev: none. Judge models that ran out of free daily quota during the run: none. The judge was not used in this run.
- Real API calls in this run: **31** (31 answering, 0 judge). By node: {'router': 11, 'sql_agent.explain': 10, 'sql_agent.write': 10}. Failed calls: 2.
- Questions: heldout 0 ({'data': 0, 'policy': 0, 'both': 0, 'out_of_scope': 0}), dev 10 ({'data': 15, 'policy': 20, 'both': 1, 'out_of_scope': 1}).

**Held-out is the headline. Dev numbers are optimistic because the system was tuned on those questions.**

#### Gates (must pass)

| Gate | Result | Detail |
|---|---|---|
| No unsupported number or citation in a final answer (verifier re-run on every answer) | PASS | heldout 0/0, dev 10/10 |
| All out of scope questions refused | PASS | heldout 0/0, dev 0/0 |
| SQL guardrail blocks every unsafe query | PASS | 51/51 unsafe queries blocked (`evals/unsafe_sql.json`, no model involved) |

Gate 1 uses the verifier's own rules, so it shows the verifier output is stable. It does not prove an answer is true.
Violating ids: heldout none, dev none. Gate 2 misses: heldout none, dev none.

#### Observations

| Metric | Held-out | Dev |
|---|---|---|
| Router accuracy (route matches expected) | n/a | 10/10 = 100% |
| Trajectory accuracy (exact node sequence) | n/a | 10/10 = 100% |
| SQL execution accuracy (data and both questions, 0.5% tolerance) | n/a | 10/10 = 100% |
| Retrieval recall@5 (policy and both questions, end to end) | n/a | n/a |
| Not-covered questions answered "does not cover this" (policy docs lack the answer) | n/a | n/a |
| Refusal rate on out of scope questions | n/a | n/a |
| Judge faithfulness, mean 1 to 5 | n/a | n/a |
| Judge answer relevance, mean 1 to 5 | n/a | n/a |
| Judge says faithful (faithfulness 4 or 5) | None | None |
| Answers the verifier changed | 0/0 | 1/10 |
| Sentences removed (number / citation) | 0 / 0 | 3 / 0 |
| Median wall time per question (s, includes pacing gaps) | n/a | 13.3 |
| Answering cost per question (USD, paid-tier list prices, judge excluded) | n/a | $0.000867 |

Most common wrong paths (both splits):

- none

Gold queries wrongly blocked by the SQL guardrail: 0 of 26.

Judge calibration (agreement with human labels): **not computed yet.** It needs `evals/human_labels.jsonl` to be filled in by hand.
<!-- END dev_after_fixes -->


## Dev fixes and eval v2: status (read this)

**Eval v2 has NOT been run.** The free-tier daily quota of the answering model ran out on 2026-10-25. Eval v1 stays frozen and untouched.

What was done:

- **Eval v2 questions (committed first).** 20 fresh held-out questions in `evals/questions_v2.jsonl` (6 data, 6 policy, 4 both, 4 out of scope), committed in `da8dea57bf1a1c72b71afa0162abd9abc8c0f970` before any fix and before any run. They are meant to be harder than v1: casual wording, a subquery (orders over 500 pounds), a derived number (10 percent of a 1,500 pound order), two questions the policy documents do not cover (opening hours, instalments), and boundary out of scope questions. I wrote them knowing the dev failures, but without running them and without using any of their content in a fix. Gold answers for data questions come from running gold SQL on the database.
- **Router fix (dev only).** The router prompt now says to decide by what is asked (how many or which means data, what a rule says means policy, both only for a figure and a rule) and has 5 generic examples. None of the examples is an eval question, but one is a close paraphrase of the dev question p19, so p19 is no longer a clean test of this fix.
- **RAG prompt fix (dev only).** The rule that made the model add "for a retail customer" is replaced by: use the passage's own words for who a rule applies to, and never add a customer type that neither the question nor the passage mentions.
- **Query rewrite: built, tested, NOT adopted.** I fixed the adoption rule before running it: adopt only if dev MRR@5 rises by at least 0.03 and recall@5 does not drop. Result on the 20 dev policy questions: baseline recall 0.95 and MRR 0.850, rewritten query alone 0.90 and 0.739, original plus rewritten query fused 0.95 and 0.877. The gain of 0.027 is below the rule and p10 is still missed, so `QUERY_REWRITE` stays off. The p10 retrieval problem is not fixed.
- **Not verified.** I re-ran the dev set after the fixes with the judge off (`python evals/run_evals.py --split dev --no-judge --label dev_after_fixes`). It stopped on the daily quota after 10 of 37 questions, all of them data questions. Those 10 say nothing about the fixes. Whether p07 and p19 are now routed to `policy`, and whether the "for a retail customer" habit is gone, is **unmeasured**. The fixes are covered only by unit tests of the prompts and the plumbing.
- **Exit code for incomplete runs.** If a run stops early (for example on a quota), `run_evals.py` now exits with code 2, so the nightly job fails instead of passing on the few questions that finished. A failed gate still exits with 1.
- Tests: `pytest` gives 130 passed.

To finish (after the quota resets at about 12:30 IST on 2026-10-26, midnight US Pacific):

```bash
python evals/run_evals.py --split dev --no-judge --label dev_after_fixes --resume run-dev-after-fixes
python evals/run_evals.py --questions evals/questions_v2.jsonl --label eval_v2 --questions-commit da8dea57bf1a1c72b71afa0162abd9abc8c0f970 --resume run-v2
python evals/token_report.py --records evals/runs/caching-stable_first.jsonl evals/runs/run-v2/records.jsonl
```

Quota facts measured today: `gemini-3.5-flash-lite` allowed about 440 successful calls in the quota day before returning a daily 429. `gemini-3.6-flash` allowed 17. The v2 run needs about 80 answering calls plus about 16 judge calls. The nightly job needs about 30 calls a night, so on the free tier it can fail on quota on a day when the budget was spent on experiments.


## Phase 6: API, UI and deployment (what was tested and what was not)

What was built: a FastAPI app (`POST /ask`, `GET /health`, `GET /limits`, `GET /examples`), a Streamlit chat UI that shows route, trajectory, SQL, citations with passages, the verified flag, tokens, cost and latency for every answer, usage limits (5 live questions per visitor and 60 per day for the whole demo by default, counted per US Pacific day, with an automatic pause if Gemini reports its daily quota is used up), 20 pre-computed answers from real eval-v1 runs that cost no quota, a Dockerfile, and a deployment guide (`docs/DEPLOY.md`). `pytest` gives 155 passed.

Checks that I ran:

- **Prices.** I compared all three rows of `config.PRICES` with the raw HTML of the official Gemini pricing page on 2026-10-25. They matched (gemini-3.5-flash-lite input 0.30, output 2.50, cached 0.03; gemini-3.6-flash and gemini-3.8-flash 0.75, 3.75, 0.075 until 2026-12-31). The same page says context caching is "Not available" on the free tier for gemini-3.5-flash-lite, which agrees with the explicit-caching result in Phase 5.
- **Database rebuilt from parquet.** The sales tables are committed as parquet files (7.4 MB). A database rebuilt from them gives the stored gold answer for all 36 gold queries of eval v1 and v2 (`tests/test_app_and_deploy_data.py`).
- **Clean checkout.** I cloned the committed repository into an empty folder (no database, no vector index, no `.env`) and started it with the real `scripts/start.sh`. It rebuilt the database from parquet and the vector index from the PDFs in about 40 seconds (the embedding model was already in my local cache, so a fresh host will also spend time downloading it). The API answered `/health` with the database and index present. `/examples` returned 20 examples, `/limits` showed 5 and 60, and the Streamlit health check returned ok. `start.sh` and the Dockerfile have no carriage-return characters (checked, after adding a `.gitattributes` rule).
- **Real error paths.** (1) With my shell missing the antivirus certificate settings, a live question returned a clear 502 `model_error` and the request was refunded (the session still had 5 questions). (2) With the settings, a live question hit the real Gemini daily quota: the API returned a clear 503 `model_quota_exhausted`, the demo paused itself, and a second visitor got a 429 with reason `paused` and the reset time (2026-10-26 07:00 UTC). The success path of a live question was not run in this test, because the quota was already used up. That path is covered by mocked tests and by the earlier phases.
- **The UI in a real browser.** I opened the Streamlit app in a browser, opened pre-computed examples and checked the details panel. This found two display problems (large metric tiles cut off the numbers, long SQL did not wrap), which I fixed.
- **Memory.** The agent graph alone used about 220 MB. After the first policy search (embedding model loaded, Chroma opened) the process used about 624 MB, and it stayed at 624 MB after four searches. An idle Streamlit server used about 71 MB. So the agent inside Streamlit needs roughly 700 MB in my measurement. Streamlit Community Cloud documents roughly 690 MB minimum and 2.7 GB maximum, so this is close to the minimum.
- **Slow imports.** On my machine `import sentence_transformers` takes about 35 seconds (antivirus scanning), so a cold start is slow. Hosts will differ.

Not tested: building the Docker image (Docker Desktop was not running), the Streamlit Community Cloud deployment, the GitHub Actions workflow on GitHub (including a manual `workflow_dispatch` run), and Langfuse against a live account. Nothing is deployed yet, so there is no public URL.
