### Evaluation dev_after_fixes (run `run-dev-after-fixes`)

Status: **COMPLETE**. Command: `python evals/run_evals.py` with questions from `questions.jsonl`. These held-out questions were committed before any run, in commit `fc2973aff48267cec53ede2965e2f8e6c8f708f0`.

- Answering model for every component: `gemini-3.5-flash-lite` (router low, SQL low, RAG medium, combiner medium thinking). LLM cache: **off**.
- Judge: no items were judged. By split: heldout: none; dev: none. Judge models that ran out of free daily quota during the run: none. The judge was not used in this run.
- Real API calls in this run: **92** (92 answering, 0 judge). By node: {'combiner': 1, 'rag_agent': 21, 'router': 38, 'sql_agent.explain': 16, 'sql_agent.write': 16}. Failed calls: 2.
- Questions: heldout 0 ({'data': 0, 'policy': 0, 'both': 0, 'out_of_scope': 0}), dev 37 ({'data': 15, 'policy': 20, 'both': 1, 'out_of_scope': 1}).

**Held-out is the headline. Dev numbers are optimistic because the system was tuned on those questions.**

#### Gates (must pass)

| Gate | Result | Detail |
|---|---|---|
| No unsupported number or citation in a final answer (verifier re-run on every answer) | PASS | heldout 0/0, dev 36/36 |
| All out of scope questions refused | PASS | heldout 0/0, dev 1/1 |
| SQL guardrail blocks every unsafe query | PASS | 51/51 unsafe queries blocked (`evals/unsafe_sql.json`, no model involved) |

Gate 1 uses the verifier's own rules, so it shows the verifier output is stable. It does not prove an answer is true.
Violating ids: heldout none, dev none. Gate 2 misses: heldout none, dev none.

#### Observations

| Metric | Held-out | Dev |
|---|---|---|
| Router accuracy (route matches expected) | n/a | 37/37 = 100% |
| Trajectory accuracy (exact node sequence) | n/a | 37/37 = 100% |
| SQL execution accuracy (data and both questions, 0.5% tolerance) | n/a | 15/16 = 94% |
| Retrieval recall@5 (policy and both questions, end to end) | n/a | 20/21 = 95% |
| Not-covered questions answered "does not cover this" (policy docs lack the answer) | n/a | n/a |
| Refusal rate on out of scope questions | n/a | 1/1 = 100% |
| Judge faithfulness, mean 1 to 5 | n/a | n/a |
| Judge answer relevance, mean 1 to 5 | n/a | n/a |
| Judge says faithful (faithfulness 4 or 5) | None | None |
| Answers the verifier changed | 0/0 | 1/36 |
| Sentences removed (number / citation) | 0 / 0 | 3 / 0 |
| Median wall time per question (s, includes pacing gaps) | n/a | 29.3 |
| Answering cost per question (USD, paid-tier list prices, judge excluded) | n/a | $0.001650 |

Most common wrong paths (both splits):

- none

Gold queries wrongly blocked by the SQL guardrail: 0 of 26.

Judge calibration (agreement with human labels): **not computed yet.** It needs `evals/human_labels.jsonl` to be filled in by hand.
