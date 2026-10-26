### Evaluation eval_v2 (run `run-v2`)

Status: **COMPLETE**. Command: `python evals/run_evals.py` with questions from `questions_v2.jsonl`. These held-out questions were committed before any run, in commit `73ad350`.

- Answering model for every component: `gemini-3.5-flash-lite` (router low, SQL low, RAG medium, combiner medium thinking). LLM cache: **off**.
- Judge: `gemini-3.6-flash` scored 15. By split: heldout: 15 by `gemini-3.6-flash`; dev: 0 by `gemini-3.6-flash`. Judge models that ran out of free daily quota during the run: none. The judge was a different model from the answering model.
- Real API calls in this run: **68** (53 answering, 15 judge). By node: {'combiner': 4, 'judge': 15, 'rag_agent': 9, 'router': 20, 'sql_agent.explain': 10, 'sql_agent.write': 10}. Failed calls: 0.
- Questions: heldout 20 ({'data': 6, 'policy': 6, 'both': 4, 'out_of_scope': 4}), dev 0 ({'data': 0, 'policy': 0, 'both': 0, 'out_of_scope': 0}).

**Held-out is the headline. Dev numbers are optimistic because the system was tuned on those questions.**

#### Gates (must pass)

| Gate | Result | Detail |
|---|---|---|
| No unsupported number or citation in a final answer (verifier re-run on every answer) | PASS | heldout 15/15, dev 0/0 |
| All out of scope questions refused | PASS | heldout 4/4, dev 0/0 |
| SQL guardrail blocks every unsafe query | PASS | 51/51 unsafe queries blocked (`evals/unsafe_sql.json`, no model involved) |

Gate 1 uses the verifier's own rules, so it shows the verifier output is stable. It does not prove an answer is true.
Violating ids: heldout none, dev none. Gate 2 misses: heldout none, dev none.

#### Observations

| Metric | Held-out | Dev |
|---|---|---|
| Router accuracy (route matches expected) | 19/20 = 95% (95% CI low 76%) | n/a |
| Trajectory accuracy (exact node sequence) | 19/20 = 95% (95% CI low 76%) | n/a |
| SQL execution accuracy (data and both questions, 0.5% tolerance) | 10/10 = 100% (95% CI low 72%) | n/a |
| Retrieval recall@5 (policy and both questions, end to end) | 7/8 = 88% (95% CI low 53%) | n/a |
| Not-covered questions answered "does not cover this" (policy docs lack the answer) | 1/2 = 50% (95% CI low 9%) | n/a |
| Refusal rate on out of scope questions | 4/4 = 100% (95% CI low 51%) | n/a |
| Judge faithfulness, mean 1 to 5 | 5.00 (n=15) | n/a |
| Judge answer relevance, mean 1 to 5 | 5.00 (n=15) | n/a |
| Judge says faithful (faithfulness 4 or 5) | 100% | None |
| Answers the verifier changed | 0/15 | 0/0 |
| Sentences removed (number / citation) | 0 / 0 | 0 / 0 |
| Median wall time per question (s, includes pacing gaps) | 41.8 | n/a |
| Answering cost per question (USD, paid-tier list prices, judge excluded) | $0.001715 | n/a |

Most common wrong paths (both splits):

- 1x expected router > rag_agent > combiner > verifier | got router > refuse

Gold queries wrongly blocked by the SQL guardrail: 0 of 10.

Judge calibration (agreement with human labels): **not computed yet.** It needs `evals/human_labels.jsonl` to be filled in by hand.
