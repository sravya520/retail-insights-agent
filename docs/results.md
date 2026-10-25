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

- **Example answers were generated in a second run, after two fixes.** The first run (reranker on, original prompt) said "The policy documents do not cover this" for the Spain shipping question. Cause: the reranker pushed Shipping §3.2 out of the top 5, and the model did not link Spain to "EU". Fixes: the examples now use plain hybrid search (the best method in the eval), and the prompt lets the model map a country to a region named in the policy. The first run is not kept. I did not fix the model changing its answer on a rerun, so the examples show one run only.
- The configured model `gemini-2.5-flash` returned a 404 (no longer available). The model is now `gemini-3.8-flash`, set in `src/ria/config.py`. This model thinks by default, and single calls took from about 5 to 40 seconds in my runs, so latency is a cost to watch in Phase 5.
- The personalised mug example is correct but wordy. It adds a wholesale case that the question did not ask for, because the prompt tells the model to give both rules when the customer type is not stated.
- Question p10 ("My item stopped working after 3 months...") is missed by every method. The gold clause is Warranty 5.2.
- The reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) did not improve recall and lowered MRR@5 from 0.850 to 0.771 on this set. With only 20 questions, one question moving one rank changes MRR by a visible amount, so treat small differences as noise.
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
| Hybrid + reranker | 0.95 | 0.771 |

Questions with no gold clause in the top 5:

- BM25 only: p03, p05, p06, p10, p14, p19
- Vector only: p10
- Hybrid (RRF k=60): p10
- Hybrid + reranker: p10
<!-- END phase1_retrieval -->

<!-- BEGIN phase1_examples -->
### Example answers

Command: `python evals/rag_examples.py`

**Q:** Can a wholesale customer return a damaged item after 40 days?

**A:** Wholesale customers must report damaged goods within 48 hours of delivery [Returns and Refunds §7.4, p.2]. Claims made after 48 hours may be refused [Returns and Refunds §7.4, p.2].

Citations matched to retrieved chunks: 1. Not matched: 0.

---

**Q:** I bought a personalised mug on 5 December. Can I return it in January because I changed my mind?

**A:** No, you cannot return the personalised mug because you changed your mind [Returns and Refunds §3.1, p.1]. Items that are engraved, printed or made to order cannot be returned for change of mind [Returns and Refunds §3.1, p.1]. 

Whether your purchase was retail or wholesale, personalised items cannot be returned for change of mind [Returns and Refunds §3.1, p.1]:
* For retail orders, items delivered between 1 November and 24 December may normally be returned until 31 January of the following year, but this does not apply to personalised items returned for change of mind [Returns and Refunds §2.3, p.1] [Returns and Refunds §3.1, p.1].
* For wholesale orders, change of mind returns must be made within 14 days of delivery, but personalised items are likewise excluded from change of mind returns [Returns and Refunds §3.1, p.1] [Wholesale and B2B Terms §6.1, p.2].

Citations matched to retrieved chunks: 3. Not matched: 0.

---

**Q:** How much does it cost to ship a £150 order to Spain, and who pays customs charges?

**A:** Standard delivery to Spain (which is in the EU) is free because the order is over £120 [Shipping and Delivery §3.1, p.1]. The customer is responsible for any import duties, taxes, or customs fees [Shipping and Delivery §3.2, p.1].

Citations matched to retrieved chunks: 2. Not matched: 0.

---

**Q:** Do you offer a price match against other shops?

**A:** The policy documents do not cover this.

Citations matched to retrieved chunks: 0. Not matched: 0.
<!-- END phase1_examples -->
