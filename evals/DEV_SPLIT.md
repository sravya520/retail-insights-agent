# Dev set versus held-out set

Questions used while building or tuning prompts and retrieval are called **dev**. Numbers on dev
questions can be optimistic. The Phase 4 eval must also contain **held-out** questions that were
never used to tune anything. Every eval file has a `split` field (`dev` or `heldout`).

## Dev questions so far

- `evals/policy_questions.jsonl` p01 to p20: all dev. I saw their retrieval results (and the p10 miss).
- Three RAG example questions in `evals/rag_examples.py`: the damaged wholesale item after 40 days (same as p05),
  the personalised mug bought on 5 December, and the £150 order to Spain. The mug and Spain questions were used to
  tune the RAG prompt. The price match question was used as an out-of-scope check.
- `evals/sql_questions.jsonl` q01 to q15 (Phase 2): all dev, because the SQL prompt and schema text are tuned on them.

## Rule for Phase 4

Phase 4 adds new held-out policy, data, mixed and out-of-scope questions, written without looking at model output.
Reported metrics are split into dev and held-out. The held-out numbers are the ones to trust.

## Status after Phase 4

`evals/questions.jsonl` holds all 57 questions with a `split` field. Held-out: h01 to h20 (committed in
`fc2973aff48267cec53ede2965e2f8e6c8f708f0` before any run). Dev: q01 to q15, p01 to p20, b01 and o01.
No prompt, retrieval setting or guardrail was changed after the held-out results were seen.
