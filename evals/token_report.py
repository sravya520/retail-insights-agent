"""Token economics report: tokens and cost per route, cost per 1,000 queries, and the share of tokens spent on the
schema and on retrieved policy context.

Reads saved request records (they hold per-call token counts and cost). No answering calls are made. Component sizes
are measured with the Gemini count_tokens endpoint (it does not generate text).

Run: python evals/token_report.py [--records evals/runs/caching-stable_first.jsonl evals/runs/run-v2/records.jsonl]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ria import config  # noqa: E402
from ria import token_stats as ts  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.sql_agent import FEW_SHOT, SYSTEM_PROMPT as SQL_SYSTEM_PROMPT  # noqa: E402
from ria.sql_db import SCHEMA_DESCRIPTION  # noqa: E402

DEFAULT_RECORDS = [ROOT / "evals" / "runs" / "caching-stable_first.jsonl"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()] if path.exists() else []


def context_text(chunks: list[dict]) -> str:
    """The retrieved passages exactly as the RAG prompt formats them."""
    return "\n\n".join(f"Passage {i}: [{c['doc']} §{c['clause_id']}, p.{c['page']}]\n{c['text']}" for i, c in enumerate(chunks, start=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", nargs="*", type=Path, default=DEFAULT_RECORDS)
    args = parser.parse_args()

    records = [r for p in args.records for r in read_jsonl(p) if r.get("usage")]
    if not records:
        sys.exit("No records with usage data found. Run evals/caching_experiment.py first.")

    from google import genai
    client = genai.Client(api_key=config.GOOGLE_API_KEY)

    def count(text: str) -> int:
        return client.models.count_tokens(model=config.GEMINI_MODEL, contents=text).total_tokens

    sizes = {"schema": count(SCHEMA_DESCRIPTION), "few_shot": count(FEW_SHOT), "sql_system": count(SQL_SYSTEM_PROMPT)}
    contexts = {r["id"]: count(context_text(r["rag"]["chunks"])) for r in records if r.get("rag") and r["rag"].get("chunks")}
    shares = ts.component_shares(records, sizes, contexts)
    routes = ts.by_route(records)
    overall = ts.summarise(records)
    nodes = ts.by_node(records)

    route_rows = "\n".join(
        f"| {route} | {s['requests']} | {s['llm_calls'] / s['requests']:.1f} | {s['avg_input_per_query']:.0f} | {s['avg_output_per_query']:.0f} | "
        f"{s['avg_input_per_query'] + s['avg_output_per_query']:.0f} | ${s['cost_per_query']:.6f} | ${s['cost_per_1000']:.3f} |"
        for route, s in routes.items())
    node_rows = "\n".join(f"| {n} | {v['calls']} | {v['avg_input']:.0f} | {v['avg_output']:.0f} |" for n, v in nodes.items())

    def share_row(label: str, key: str) -> str:
        s = shares[key]
        return f"| {label} | {s['tokens']} | {s['share_of_input']:.1%} | {s['share_of_all_tokens']:.1%} |"

    sources = ", ".join(f"`{p.name}`" for p in args.records)
    mix = ", ".join(f"{route} {s['requests']}" for route, s in routes.items())
    out_share = overall["output_tokens"] / (overall["input_tokens"] + overall["output_tokens"])
    update_section("phase5_tokens", f"""### Token economics (per route)

Command: `python evals/token_report.py`. Source records: {sources} ({len(records)} requests: {mix}). Model: `{config.GEMINI_MODEL}` for every component,
production prompt layout, LLM response cache off. The judge is not included (it is an evaluation cost, not a serving cost).
Cost uses `config.PRICES` (paid-tier list prices from the Gemini pricing page; thinking tokens are billed as output). The free tier bills nothing.
Route groups are small (4 to 12 requests), so treat the averages as rough.

| Route | Requests | LLM calls per query | Avg input tokens | Avg output tokens | Avg total tokens | Cost per query | Cost per 1,000 queries |
|---|---|---|---|---|---|---|---|
{route_rows}
| **all (this mix)** | {overall['requests']} | {overall['llm_calls'] / overall['requests']:.1f} | {overall['avg_input_per_query']:.0f} | {overall['avg_output_per_query']:.0f} | {overall['avg_input_per_query'] + overall['avg_output_per_query']:.0f} | ${overall['cost_per_query']:.6f} | ${overall['cost_per_1000']:.3f} |

Output tokens (which include thinking tokens) are {out_share:.0%} of all tokens in these requests, and cost {config.PRICES[config.GEMINI_MODEL]['output'] / config.PRICES[config.GEMINI_MODEL]['input']:.1f} times more per token than input tokens.

Average tokens per call, by node:

| Node | Calls | Avg input | Avg output |
|---|---|---|---|
{node_rows}

Share of tokens spent on the schema and on retrieved context. Fixed prompt parts were counted with the Gemini `count_tokens` endpoint
(schema description {sizes['schema']} tokens, few-shot examples {sizes['few_shot']}, whole SQL system prompt {sizes['sql_system']}). The retrieved context is the
policy passages sent to the RAG agent, counted per request. Counts are approximate because parts are counted separately from the whole prompt.
Total input tokens in these requests: {shares['total_input_tokens']}. Total tokens including output: {shares['total_tokens']}.

| Part of the prompt | Tokens | Share of input tokens | Share of all tokens (input + output) |
|---|---|---|---|
{share_row('Schema description (sent in every SQL write call)', 'schema_description')}
{share_row('Few-shot SQL examples', 'few_shot_examples')}
{share_row('Other SQL instructions and rules', 'other_sql_instructions')}
{share_row('Retrieved policy context (5 passages in every RAG call)', 'retrieved_policy_context')}
""")
    print("token report written")


if __name__ == "__main__":
    main()
