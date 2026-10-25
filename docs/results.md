# Results

Every number below was produced by the command shown next to it. Sections between the BEGIN and END markers are rewritten by the scripts.

## Which model produced which result

| Result | Model | Notes |
|---|---|---|
| Retrieval comparison (Phase 1) | No Gemini model. Embeddings `BAAI/bge-small-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2` | Runs locally. |
| RAG example answers (Phase 1) | `gemini-3.5-flash-lite`, RAG thinking `medium` | Run before the cache existed. |
| SQL accuracy (Phase 2) | `gemini-3.5-flash-lite`, SQL thinking `low` | Cache off. |
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
