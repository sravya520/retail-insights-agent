"""Judge calibration: how often does the LLM judge agree with your hand labels?

Refuses to run until every line of evals/human_labels.jsonl has "faithful" set to true or false.
The judge counts as saying "faithful" when its faithfulness score is 4 or 5 (fixed before any labels existed).

Run: python evals/judge_agreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria.eval_metrics import judge_agreement  # noqa: E402
from ria.results_md import update_section  # noqa: E402

LABELS = ROOT / "evals" / "human_labels.jsonl"


def main() -> None:
    labels = [json.loads(l) for l in LABELS.read_text(encoding="utf-8").splitlines() if l.strip()]
    missing = [l["id"] for l in labels if not isinstance(l.get("faithful"), bool)]
    if missing:
        print(f"{len(missing)} of {len(labels)} labels are still empty (ids: {missing}). Set faithful to true or false, then rerun.")
        sys.exit(1)
    run_id = labels[0]["run_id"]
    records = {}
    for l in (ROOT / "evals" / "runs" / run_id / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            records[r["id"]] = r
    judge_faithful = {i: r["judge"]["faithfulness"] >= 4 for i, r in records.items() if r.get("judge")}
    res = judge_agreement(labels, judge_faithful)
    models = sorted({records[l["id"]]["judge"]["model"] for l in labels})
    human_not = sum(not l["faithful"] for l in labels)
    kappa = f"{res['kappa']:.2f}" if res["kappa"] is not None else "n/a (one class only)"
    text = f"""### Judge calibration (20 hand-labelled answers, run `{run_id}`)

Command: `python evals/judge_agreement.py`. Judge model(s) that scored these answers: {', '.join(f'`{m}`' for m in models)}.

- Agreement between the judge (faithfulness 4 or 5 = faithful) and the human labels: **{res['agreement']:.0%}** ({res['both_faithful'] + res['both_not_faithful']} of {res['n']}).
- Cohen's kappa: {kappa}.
- Human said not faithful on {human_not} answers. Judge said faithful when the human said not: {res['judge_faithful_human_not']}. Judge said not faithful when the human said faithful: {res['judge_not_human_faithful']}.
- Disagreeing ids: {res['disagreements'] or 'none'}.
"""
    update_section("phase4_judge_calibration", text)
    print(text)


if __name__ == "__main__":
    main()
