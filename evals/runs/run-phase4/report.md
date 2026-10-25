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
