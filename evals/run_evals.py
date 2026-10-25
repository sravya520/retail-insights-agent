"""Phase 4 evaluation harness.

Runs every question in evals/questions.jsonl through the full graph (held-out first, then dev), judges the answers,
computes metrics per split, checks the gates, and writes docs/results.md.

  * One model for all components (config.GEMINI_MODEL). The LLM cache is OFF. Real calls are counted.
  * Each question is saved to evals/runs/<run_id>/records.jsonl as soon as it finishes.
  * If the daily quota runs out, the run stops cleanly, saves what it has and reports how far it got.
    Resume with:  python evals/run_evals.py --resume <run_id>

Run: python evals/run_evals.py            (new run)
     python evals/run_evals.py --report <run_id>   (recompute the report from saved records, no API calls)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria import eval_metrics as m  # noqa: E402
from ria.graph import ask, build_graph, default_services  # noqa: E402
from ria.judge import JudgePool, evidence_text  # noqa: E402
from ria.llm import LLMError  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.sql_db import connect_readonly, get_schema  # noqa: E402
from ria.tracing import LLM_CALLS_PATH  # noqa: E402

RUNS = ROOT / "evals" / "runs"
QUESTIONS = ROOT / "evals" / "questions.jsonl"
HELDOUT_COMMIT = "fc2973aff48267cec53ede2965e2f8e6c8f708f0"


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def call_log_size() -> int:
    return len(LLM_CALLS_PATH.read_text(encoding="utf-8-sig").splitlines()) if LLM_CALLS_PATH.exists() else 0


def calls_since(offset: int) -> list[dict]:
    lines = LLM_CALLS_PATH.read_text(encoding="utf-8-sig").splitlines()[offset:] if LLM_CALLS_PATH.exists() else []
    return [json.loads(l) for l in lines if l.strip()]


def is_daily_quota(exc: BaseException) -> bool:
    return "PerDay" in str(exc) or "per day" in str(exc)


def run_one(app, pool: JudgePool, q: dict, run_id: str) -> dict:
    offset = call_log_size()
    rec = {k: q.get(k) for k in ("id", "split", "question", "expected_route", "expected_trajectory", "ordered",
                                 "gold_answer", "gold_citations", "gold_text")}
    started = time.perf_counter()
    try:
        out = ask(app, q["question"], thread_id=f"{run_id}-{q['id']}")
    except LLMError as exc:
        if is_daily_quota(exc):
            raise
        rec.update(error=str(exc)[:300], route=None, trajectory=None, answer="", verified=None, removed=[])
        rec["latency_s"] = round(time.perf_counter() - started, 3)
        rec["calls"] = len(calls_since(offset))
        return rec
    sql, rag = out.get("sql"), out.get("rag")
    rec.update(
        standalone_question=out.get("standalone_question"), route=out.get("route"), trajectory=out.get("trajectory"),
        answer=out.get("answer", ""), draft=out.get("draft", ""), verified=out.get("verified"),
        removed=(out.get("verifier") or {}).get("removed", []), latency_s=out["latency_s"], error=None,
        sql={k: sql[k] for k in ("sql", "columns", "rows", "error", "repaired")} if sql else None,
        rag={"chunks": rag["chunks"], "citations": rag["citations"]} if rag else None)
    judge_calls_offset = call_log_size()
    if out.get("route") != "out_of_scope" and rec["answer"]:
        try:
            res = pool.score(q["question"], rec["answer"], evidence_text(rec["sql"], (rag or {}).get("chunks")))
            rec["judge"] = {"faithfulness": res.faithfulness, "relevance": res.relevance,
                            "unsupported_claims": res.unsupported_claims, "model": res.model}
        except LLMError as exc:
            if is_daily_quota(exc) or "No judge model" in str(exc):
                raise
            rec["judge"], rec["judge_error"] = None, str(exc)[:200]
    calls = calls_since(offset)
    rec["calls"] = len(calls)
    rec["judge_calls"] = len(calls_since(judge_calls_offset))
    return rec


def fmt(num: int, den: int) -> str:
    return f"{num}/{den} = {num / den:.0%}" if den else "n/a"


def fmt_ratio(pair: tuple, bound: bool = False) -> str:
    text = fmt(pair[0], pair[1])
    if bound and pair[1]:
        text += f" (95% CI low {m.wilson_lower(pair[0], pair[1]):.0%})"
    return text


def fmt_latency(records: list, bound: bool = False) -> str:
    value = m.median_latency(records)
    return f"{value:.1f}" if value else "n/a"


def build_report(run_id: str, records: list[dict], meta: dict) -> str:
    splits = {"heldout": [r for r in records if r["split"] == "heldout"], "dev": [r for r in records if r["split"] == "dev"]}
    schema = get_schema(connect_readonly())
    unsafe = json.loads((ROOT / "evals" / "unsafe_sql.json").read_text(encoding="utf-8"))
    gold_sqls = [q["gold_sql"] for q in read_jsonl(QUESTIONS) if q.get("gold_sql")]

    def row(label: str, fn) -> str:
        return f"| {label} | {fn(splits['heldout'], True)} | {fn(splits['dev'], False)} |"

    traj = {k: m.trajectory_accuracy(v) for k, v in splits.items()}
    wrong = traj["heldout"][2] + traj["dev"][2]
    wrong_lines = "\n".join(f"- {n}x {path}" for path, n in wrong.most_common(8)) or "- none"
    jm = {k: m.judge_means(v) for k, v in splits.items()}
    vs = {k: m.verifier_stats(v) for k, v in splits.items()}

    def jfmt(d, key):
        return f"{d[key]:.2f} (n={d['n']})" if d[key] is not None else "n/a"

    g1 = {k: m.gate_no_unsupported_content(v) for k, v in splits.items()}
    g2 = {k: m.gate_out_of_scope_refused(v) for k, v in splits.items()}
    g3 = m.gate_unsafe_sql_blocked(unsafe, schema)
    blocked_gold = m.gold_queries_not_blocked(gold_sqls, schema)

    def gate_line(name: str, bad_h: int, n_h: int, bad_d: int, n_d: int, extra: str = "") -> str:
        ok = bad_h == 0 and bad_d == 0
        return f"| {name} | {'PASS' if ok else 'FAIL'} | heldout {n_h - bad_h}/{n_h}, dev {n_d - bad_d}/{n_d}{extra} |"

    judge_models = sorted({r["judge"]["model"] for r in records if r.get("judge")})
    judge_counts = {mod: sum(1 for r in records if r.get("judge") and r["judge"]["model"] == mod) for mod in judge_models}
    judge_by_split = "; ".join(
        f"{s}: " + (", ".join(f"{sum(1 for r in v if r.get('judge') and r['judge']['model'] == mod)} by `{mod}`" for mod in judge_models) or "none")
        for s, v in splits.items())
    judge_note = ", ".join(f"`{mod}` scored {n}" for mod, n in judge_counts.items()) or "no items were judged"
    self_judged = config.GEMINI_MODEL in judge_counts
    dead = ", ".join(f"`{d}`" for d in meta.get("judge_models_dead", [])) or "none"
    calls = meta.get("calls", {})
    status = "COMPLETE" if meta["completed"] == meta["total"] else f"PARTIAL: {meta['completed']} of {meta['total']} questions finished ({meta['stop_reason']})"

    return f"""### Phase 4 evaluation (run `{run_id}`)

Status: **{status}**. Command: `python evals/run_evals.py`. Held-out questions were committed before any run in commit `{HELDOUT_COMMIT}`.

- Answering model for every component: `{config.GEMINI_MODEL}` (router low, SQL low, RAG medium, combiner medium thinking). LLM cache: **off**.
- Judge: {judge_note}. By split: {judge_by_split}. Judge models that ran out of free daily quota during the run: {dead}. {"Some items were self-judged (the answering model scored its own answers), which can inflate scores. See the limits below." if self_judged else "The judge was a different model from the answering model."}
- Real API calls in this run: **{calls.get('total', 0)}** ({calls.get('answering', 0)} answering, {calls.get('judge', 0)} judge). By node: {calls.get('by_node', {})}. Failed calls: {calls.get('failed', 0)}.
- Questions: heldout {len(splits['heldout'])} ({meta['route_counts']['heldout']}), dev {len(splits['dev'])} ({meta['route_counts']['dev']}).

**Held-out is the headline. Dev numbers are optimistic because the system was tuned on those questions.**

#### Gates (must pass)

| Gate | Result | Detail |
|---|---|---|
{gate_line('No unsupported number or citation in a final answer (verifier re-run on every answer)', g1['heldout'][0], g1['heldout'][1], g1['dev'][0], g1['dev'][1])}
{gate_line('All out of scope questions refused', g2['heldout'][0], g2['heldout'][1], g2['dev'][0], g2['dev'][1])}
| SQL guardrail blocks every unsafe query | {'PASS' if g3[0] == 0 else 'FAIL'} | {g3[1] - g3[0]}/{g3[1]} unsafe queries blocked (`evals/unsafe_sql.json`, no model involved) |

Gate 1 uses the verifier's own rules, so it shows the verifier output is stable. It does not prove an answer is true.
Violating ids: heldout {g1['heldout'][2] or 'none'}, dev {g1['dev'][2] or 'none'}. Gate 2 misses: heldout {g2['heldout'][2] or 'none'}, dev {g2['dev'][2] or 'none'}.

#### Observations

| Metric | Held-out | Dev |
|---|---|---|
{row('Router accuracy (route matches expected)', lambda r, b: fmt_ratio(m.router_accuracy(r), b))}
{row('Trajectory accuracy (exact node sequence)', lambda r, b: fmt_ratio(m.trajectory_accuracy(r)[:2], b))}
{row('SQL execution accuracy (data and both questions, 0.5% tolerance)', lambda r, b: fmt_ratio(m.sql_execution_accuracy(r), b))}
{row('Retrieval recall@5 (policy and both questions, end to end)', lambda r, b: fmt_ratio(m.retrieval_recall_at_5(r), b))}
{row('Refusal rate on out of scope questions', lambda r, b: fmt_ratio(m.refusal_rate(r), b))}
| Judge faithfulness, mean 1 to 5 | {jfmt(jm['heldout'], 'faithfulness')} | {jfmt(jm['dev'], 'faithfulness')} |
| Judge answer relevance, mean 1 to 5 | {jfmt(jm['heldout'], 'relevance')} | {jfmt(jm['dev'], 'relevance')} |
| Judge says faithful (faithfulness 4 or 5) | {jm['heldout']['faithful_share'] if jm['heldout']['faithful_share'] is None else format(jm['heldout']['faithful_share'], '.0%')} | {jm['dev']['faithful_share'] if jm['dev']['faithful_share'] is None else format(jm['dev']['faithful_share'], '.0%')} |
| Answers the verifier changed | {vs['heldout']['answers_changed']}/{vs['heldout']['answers_checked']} | {vs['dev']['answers_changed']}/{vs['dev']['answers_checked']} |
| Sentences removed (number / citation) | {vs['heldout']['removed_unsupported_number']} / {vs['heldout']['removed_invalid_citation']} | {vs['dev']['removed_unsupported_number']} / {vs['dev']['removed_invalid_citation']} |
{row('Median wall time per question (s, includes pacing gaps)', fmt_latency)}

Most common wrong paths (both splits):

{wrong_lines}

Gold queries wrongly blocked by the SQL guardrail: {len(blocked_gold)} of {len(gold_sqls)}.

Judge calibration (agreement with human labels): **not computed yet.** It needs `evals/human_labels.jsonl` to be filled in by hand.
"""


def summarise(run_id: str, records: list[dict], meta: dict) -> None:
    report = build_report(run_id, records, meta)
    (RUNS / run_id / "report.md").write_text(report, encoding="utf-8")
    update_section("phase4_eval", report)
    print(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", help="run id to continue")
    parser.add_argument("--report", help="rebuild the report from saved records, no API calls")
    parser.add_argument("--limit", type=int, help="only the first N questions of each split (for a smoke test)")
    args = parser.parse_args()

    config.LLM_CACHE_ENABLED = False  # final eval: cache OFF
    questions = read_jsonl(QUESTIONS)
    ordered = [q for q in questions if q["split"] == "heldout"] + [q for q in questions if q["split"] == "dev"]
    if args.limit:
        ordered = [q for s in ("heldout", "dev") for q in [x for x in ordered if x["split"] == s][:args.limit]]

    run_id = args.resume or args.report or time.strftime("run-%Y%m%d-%H%M%S")
    run_dir = RUNS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"
    meta_path = run_dir / "meta.json"
    records = read_jsonl(records_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"calls": {}, "stop_reason": "", "log_start": call_log_size()}
    meta["total"] = len(ordered)
    meta["route_counts"] = {s: {r: sum(1 for q in ordered if q["split"] == s and q["expected_route"] == r)
                                for r in ("data", "policy", "both", "out_of_scope")} for s in ("heldout", "dev")}

    if not args.report:
        done = {r["id"] for r in records}
        app = build_graph(default_services())
        pool = JudgePool(config.JUDGE_MODELS)
        try:
            for q in ordered:
                if q["id"] in done:
                    continue
                rec = run_one(app, pool, q, run_id)
                records.append(rec)
                with records_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                print(f"{q['split']:8s} {q['id']:4s} route={rec.get('route')} traj_ok={rec.get('trajectory') == q['expected_trajectory']} "
                      f"judge={(rec.get('judge') or {}).get('faithfulness')} calls={rec.get('calls')} t={rec.get('latency_s')}s", flush=True)
        except LLMError as exc:
            meta["stop_reason"] = f"stopped on error: {str(exc)[:200]}"
            print("STOPPING CLEANLY:", meta["stop_reason"], flush=True)
        meta["judge_models_dead"] = sorted(pool.dead)

    meta["completed"] = len(records)
    log = calls_since(meta["log_start"])
    real = [c for c in log if not c.get("cache_hit")]
    meta["calls"] = {"total": len(real), "judge": sum(c["node"] == "judge" for c in real),
                     "answering": sum(c["node"] != "judge" for c in real),
                     "failed": sum("error" in c for c in real),
                     "by_node": {n: sum(c["node"] == n for c in real) for n in sorted({c["node"] for c in real})}}
    meta["model"] = config.GEMINI_MODEL
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    summarise(run_id, records, meta)
    gates_failed = m.gate_unsafe_sql_blocked(json.loads((ROOT / "evals" / "unsafe_sql.json").read_text(encoding="utf-8")),
                                             get_schema(connect_readonly()))[0]
    gates_failed += m.gate_no_unsupported_content(records)[0] + m.gate_out_of_scope_refused(records)[0]
    sys.exit(1 if gates_failed else 0)


if __name__ == "__main__":
    main()
