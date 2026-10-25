"""Caching experiment: does putting the long stable prompt part first make the API cache it?

Runs the 20 held-out (v1) questions through the whole graph twice, LLM response cache OFF:
  question_first  baseline layout: question at the top, instructions, schema and examples after it
  stable_first    production layout: instructions, schema and examples first, question last
Then a probe sends the identical SQL prompt several times in a row to see if repeated prefixes are cached.
Reports cached tokens, cost per query and latency for both layouts. Only measured numbers are reported.

The baseline layout is a deliberate comparison layout, not something that ran in production before.
No judge calls. Answers are not scored here, except a sanity check that routes and SQL results did not change.

Run: python evals/caching_experiment.py          (resumes if records already exist)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria import eval_metrics as m  # noqa: E402
from ria import token_stats as ts  # noqa: E402
from ria.eval_record import record_from_state  # noqa: E402
from ria.graph import ask, build_graph, default_services  # noqa: E402
from ria.llm import LLMError, get_llm, invoke_with_retry  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.sql_agent import SYSTEM_PROMPT as SQL_SYSTEM_PROMPT  # noqa: E402

RUNS = ROOT / "evals" / "runs"
LAYOUTS = ["question_first", "stable_first"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def run_layout(app, layout: str, questions: list[dict]) -> list[dict]:
    config.PROMPT_LAYOUT = layout
    path = RUNS / f"caching-{layout}.jsonl"
    records = read_jsonl(path)
    done = {r["id"] for r in records}
    for q in questions:
        if q["id"] in done:
            continue
        out = ask(app, q["question"], thread_id=f"cache-{layout}-{q['id']}", label=f"cache-{layout}")
        rec = record_from_state(q, out)
        rec["layout"] = layout
        records.append(rec)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        print(f"{layout:15s} {q['id']} route={rec['route']} in={rec['usage']['input_tokens']} cached={rec['usage']['cached_tokens']} "
              f"cost=${rec['usage']['cost_usd']:.6f} model_s={rec['usage']['llm_latency_s']}", flush=True)
    return records


def repeat_probe(n: int = 4) -> list[dict]:
    """Send the same SQL prompt n times in a row. If prefix caching works at this prompt size, later calls show cached tokens."""
    config.PROMPT_LAYOUT = "stable_first"
    llm = get_llm("sql")
    messages = [("system", SQL_SYSTEM_PROMPT), ("human", "Question: What was the total revenue in March 2011?\nSQL:")]
    for _ in range(n):
        invoke_with_retry(llm, messages, node="cache_probe")
    # usage is read from the call log (invoke_with_retry returns a normalised message without usage)
    lines = [json.loads(l) for l in (ROOT / "logs" / "llm_calls.jsonl").read_text(encoding="utf-8-sig").splitlines() if l.strip()]
    probe = [l for l in lines if l["node"] == "cache_probe"][-n:]
    return [{"call": i + 1, "input_tokens": p["input_tokens"], "cached_tokens": p["cached_tokens"], "latency_s": p["latency_s"]}
            for i, p in enumerate(probe)]


def sanity(records: list[dict]) -> dict:
    return {"router": m.router_accuracy(records), "trajectory": m.trajectory_accuracy(records)[:2],
            "sql": m.sql_execution_accuracy(records)}


def fmt_row(label: str, s: dict) -> str:
    return (f"| {label} | {s['requests']} | {s['llm_calls']} | {s['input_tokens']} | {s['cached_tokens']} "
            f"({s['cached_share_of_input']:.1%}) | ${s['cost_per_query']:.6f} | ${s['cost_per_1000']:.3f} | "
            f"{s['median_call_latency_s']:.2f} | {s['median_query_model_latency_s']:.2f} |")


def main() -> None:
    config.LLM_CACHE_ENABLED = False
    questions = [q for q in read_jsonl(ROOT / "evals" / "questions.jsonl") if q["split"] == "heldout"]
    app = build_graph(default_services())
    calls_before = len((ROOT / "logs" / "llm_calls.jsonl").read_text(encoding="utf-8-sig").splitlines())
    results: dict[str, list[dict]] = {}
    stopped = ""
    try:
        for layout in LAYOUTS:
            results[layout] = run_layout(app, layout, questions)
        probe = repeat_probe()
    except LLMError as exc:
        stopped = f"stopped early: {str(exc)[:200]}"
        probe = []
        print("STOPPING CLEANLY:", stopped)
        for layout in LAYOUTS:
            results.setdefault(layout, read_jsonl(RUNS / f"caching-{layout}.jsonl"))
    log = [json.loads(l) for l in (ROOT / "logs" / "llm_calls.jsonl").read_text(encoding="utf-8-sig").splitlines()[calls_before:] if l.strip()]
    real_calls = sum(not c.get("cache_hit") for c in log)

    s = {layout: ts.summarise(recs) for layout, recs in results.items()}
    sane = {layout: sanity(recs) for layout, recs in results.items()}
    before, after = s.get("question_first"), s.get("stable_first")

    def delta(key: str, pct: bool = True) -> str:
        if not before or not after or not before[key]:
            return "n/a"
        d = after[key] - before[key]
        return f"{d:+.6g}" + (f" ({d / before[key]:+.1%})" if pct else "")

    probe_lines = "\n".join(f"| {p['call']} | {p['input_tokens']} | {p['cached_tokens']} | {p['latency_s']:.2f} |" for p in probe) or "| (not run) | | | |"
    node_rows = []
    for layout, recs in results.items():
        for node, v in ts.by_node(recs).items():
            node_rows.append(f"| {layout} | {node} | {v['calls']} | {v['avg_input']:.0f} | {v['avg_output']:.0f} |")

    report = f"""### Caching experiment (20 held-out v1 questions, LLM response cache off)

Command: `python evals/caching_experiment.py`. Model: `{config.GEMINI_MODEL}` for every component. {stopped or "Complete."}
Real API calls in this experiment: **{real_calls}** (including the repeat probe).

`question_first` is a comparison layout I built for this test (question first, then instructions, schema and examples).
`stable_first` is the production layout (instructions, schema and examples first, question last). The baseline never ran in production.
The two layouts ran one after the other (question_first first), so API load and time of day can differ between them.

| Layout | Queries | LLM calls | Input tokens | Cached input tokens | Cost per query | Cost per 1,000 queries | Median call latency (s) | Median model time per query (s) |
|---|---|---|---|---|---|---|---|---|
{fmt_row('question_first (before)', before) if before else ''}
{fmt_row('stable_first (after)', after) if after else ''}

Change from before to after: cached tokens {delta('cached_tokens', False)}, cost per query {delta('cost_per_query')}, median call latency {delta('median_call_latency_s')}, median model time per query {delta('median_query_model_latency_s')}.

Cost uses the price table in `config.PRICES` (paid-tier list prices, thinking tokens billed as output). Input tokens per layout differ only by a few tokens, because the same text is sent in a different order.

Average tokens per call by node and layout:

| Layout | Node | Calls | Avg input | Avg output |
|---|---|---|---|---|
{chr(10).join(node_rows)}

Repeat probe: the identical SQL prompt ({probe[0]['input_tokens'] if probe else 'n/a'} input tokens) sent {len(probe)} times in a row, production layout:

| Call | Input tokens | Cached tokens | Latency (s) |
|---|---|---|---|
{probe_lines}

Sanity check that the layout did not change behaviour (20 questions, not a quality measurement): router correct {sane.get('question_first', {}).get('router')} before and {sane.get('stable_first', {}).get('router')} after; exact trajectory {sane.get('question_first', {}).get('trajectory')} before and {sane.get('stable_first', {}).get('trajectory')} after; SQL execution accuracy {sane.get('question_first', {}).get('sql')} before and {sane.get('stable_first', {}).get('sql')} after (as correct / total).
"""
    update_section("phase5_caching", report)
    print(report)


if __name__ == "__main__":
    main()
