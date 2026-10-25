"""Write evals/human_labels.jsonl: 20 answers for you to label by hand. The judge scores are NOT in the file.

Run: python evals/make_human_labels.py <run_id>
After you fill in "faithful" (true or false) on every line, run: python evals/judge_agreement.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria.eval_metrics import label_template_row, select_for_labeling  # noqa: E402

OUT = ROOT / "evals" / "human_labels.jsonl"


def main() -> None:
    run_id = sys.argv[1]
    run_dir = ROOT / "evals" / "runs" / run_id
    records = [json.loads(l) for l in (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    chosen = select_for_labeling(records)
    OUT.write_text("\n".join(json.dumps(label_template_row(r, run_id), ensure_ascii=False) for r in chosen) + "\n", encoding="utf-8")
    # Hidden side file: what the judge said and whether the verifier changed the answer. Not shown to the human labeller.
    side = [{"id": r["id"], "split": r["split"], "route": r["route"], "verifier_changed": r.get("verified") is False,
             "judge_faithfulness": r["judge"]["faithfulness"], "judge_model": r["judge"]["model"]} for r in chosen]
    (run_dir / "label_meta.json").write_text(json.dumps(side, indent=1), encoding="utf-8")
    by_route = {r: sum(x["route"] == r for x in side) for r in ("data", "policy", "both")}
    print(f"wrote {OUT} with {len(chosen)} answers. By route: {by_route}. "
          f"Answers the verifier changed: {sum(x['verifier_changed'] for x in side)}. "
          f"Held-out: {sum(x['split'] == 'heldout' for x in side)}.")


if __name__ == "__main__":
    main()
