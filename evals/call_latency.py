"""Summarise per-call latency and token use by model and node from logs/llm_calls.jsonl.

Run: python evals/call_latency.py
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria.results_md import update_section  # noqa: E402
from ria.tracing import LLM_CALLS_PATH  # noqa: E402


def main() -> None:
    records = [json.loads(l) for l in LLM_CALLS_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r["model"], r["node"])].append(r)
    rows = []
    for (model, node), rs in sorted(groups.items()):
        ok = [r for r in rs if "error" not in r]
        lat = sorted(r["latency_s"] for r in ok) or [0.0]
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        rows.append(f"| {model} | {node} | {len(rs)} | {len(rs) - len(ok)} | {statistics.median(lat):.1f} | {p95:.1f} | "
                    f"{statistics.mean(r.get('input_tokens', 0) for r in ok) if ok else 0:.0f} | "
                    f"{statistics.mean(r.get('output_tokens', 0) for r in ok) if ok else 0:.0f} | "
                    f"{statistics.mean(r.get('reasoning_tokens', 0) for r in ok) if ok else 0:.0f} |")
    table = "\n".join(rows)
    update_section("llm_call_latency", f"""### Per-call LLM latency and tokens (from `logs/llm_calls.jsonl`)

Command: `python evals/call_latency.py`. Latency is the time inside the model call only. Pacing gaps and retry waits are not counted. Older log lines (before this was split out) include pacing, so their latency is higher than the true call time.

| Model | Node | Calls | Failed | Median s | p95 s | Avg input tokens | Avg output tokens | Avg reasoning tokens |
|---|---|---|---|---|---|---|---|---|
{table}""")
    print(table)


if __name__ == "__main__":
    main()
