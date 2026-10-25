# Results

Every number below was produced by the command shown next to it. Sections between the BEGIN and END markers are rewritten by the scripts.

## Phase 1: Documents and RAG

What was built:

- 6 synthetic policy documents for the fictional Northwind Gifts, as Markdown and PDF (2 to 3 pages each), 182 clauses in total.
- Ingestion with `pypdf`, chunked by numbered clause, with `doc`, `section`, `clause_id` and `page` kept. Embeddings with `BAAI/bge-small-en-v1.5` in Chroma.
- Hybrid retrieval (BM25 plus vector, RRF k=60) with an optional cross-encoder reranker.
- A RAG agent with a strict citation format and a citation parser that checks each citation against the retrieved chunks.
- Tests: `pytest` gives 17 passed (command: `pytest`).

What did not work or is not done:

- **Example answers are not generated yet.** There is no `GOOGLE_API_KEY` in `.env` on this machine, so the RAG agent has not been run against Gemini. The agent is tested only with a mocked LLM. Run `python evals/rag_examples.py` after adding the key. It fills the "Example answers" section below.
- Question p10 ("My item stopped working after 3 months...") is missed by every method. The gold clause is Warranty 5.2.
- The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) did not improve recall and lowered MRR@5 from 0.850 to 0.771 on this set. With only 20 questions, one question moving one rank changes MRR by a visible amount, so treat small differences as noise.
- The gold labels were written by hand before the run. They were not tuned after seeing results.
- Setup note: `pip` and the Hugging Face download failed with `CERTIFICATE_VERIFY_FAILED` because the antivirus on this machine inspects HTTPS. A CA bundle with the antivirus certificate fixed it. This is a local setup issue, not a code issue.

MRR@5 means the reciprocal rank of the first gold clause within the top 5, and 0 if it is not there. A question can list more than one acceptable gold clause.

### Example answers

Not run yet. See the note above.

<!-- BEGIN phase1_retrieval -->
### Retrieval comparison (20 policy questions)

Command: `python evals/retrieval_eval.py`

| Method | Recall@5 | MRR@5 |
|---|---|---|
| BM25 only | 0.70 | 0.608 |
| Vector only | 0.95 | 0.800 |
| Hybrid (RRF k=60) | 0.95 | 0.850 |
| Hybrid + reranker | 0.95 | 0.771 |

Questions with no gold clause in the top 5:

- BM25 only: p03, p05, p06, p10, p14, p19
- Vector only: p10
- Hybrid (RRF k=60): p10
- Hybrid + reranker: p10
<!-- END phase1_retrieval -->
