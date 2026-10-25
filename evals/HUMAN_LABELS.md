# How to fill in evals/human_labels.jsonl

The file has 20 lines. Each line is one answer the system gave. For each line, set `"faithful"` to `true` or `false`.
You can use `"notes"` for anything you want to remember. Leave every other field as it is.

## What "faithful" means

An answer is **faithful** when everything it claims is supported by the evidence shown on the same line:

- `evidence_sql`: the SQL that ran and its result rows (data answers).
- `evidence_chunks`: the policy passages that were retrieved (policy answers).

Set `"faithful": false` if the answer contains a number, rule, condition or case that the evidence does not support,
or if it contradicts the evidence. Rounding a number (for example 664431.78 to 664,431.78) is fine.
Judge only support by the evidence. Do not judge whether the answer is a good answer, or whether the evidence is the best evidence.

## Please do not look at these files until you are done

`evals/runs/run-phase4/label_meta.json` and `records.jsonl` contain the LLM judge's scores. Seeing them would bias your labels.

## Afterwards

```bash
python evals/judge_agreement.py
```

This refuses to run while any `faithful` field is empty. It then computes how often the judge agrees with you
(the judge counts as "faithful" when its faithfulness score is 4 or 5) and writes the result to `docs/results.md`.

## What is in the sample

15 of the 20 answers come from the held-out questions. The sample includes the answers the judge scored below 5
on faithfulness (the borderline ones). The verifier did not change any answer in this run, so there are no
"verifier changed" cases in the sample.
