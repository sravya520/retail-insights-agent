"""Try Gemini EXPLICIT context caching for the SQL system prompt (schema + few-shot examples).

For each model: count the prompt tokens, try to create a cached content object, and if that works, send one request
that uses it and read the cached token count. Whatever happens (including an error) is written to docs/results.md.

Run: python evals/explicit_cache_trial.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from google import genai  # noqa: E402
from google.genai import types  # noqa: E402

from ria import config  # noqa: E402
from ria.results_md import update_section  # noqa: E402
from ria.sql_agent import SYSTEM_PROMPT  # noqa: E402
from ria.sql_db import SCHEMA_DESCRIPTION  # noqa: E402

MODELS = [config.GEMINI_MODEL, "gemini-3.6-flash"]
QUESTION = "Question: What was the total revenue in March 2011?\nSQL:"


def trial(client, model: str, prompt: str = SYSTEM_PROMPT, label: str = "SQL system prompt") -> dict:
    out: dict = {"model": model, "label": label}
    out["prompt_tokens"] = client.models.count_tokens(model=model, contents=prompt).total_tokens
    try:
        cache = client.caches.create(model=model, config=types.CreateCachedContentConfig(
            system_instruction=prompt, contents=[types.Content(role="user", parts=[types.Part(text="Ready.")])], ttl="300s"))
    except Exception as exc:  # noqa: BLE001 - the error text is the result
        out["created"] = False
        out["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        return out
    out["created"] = True
    try:
        resp = client.models.generate_content(model=model, contents=QUESTION,
                                              config=types.GenerateContentConfig(cached_content=cache.name))
        usage = resp.usage_metadata
        out["input_tokens"] = usage.prompt_token_count
        out["cached_tokens"] = usage.cached_content_token_count
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
    finally:
        try:
            client.caches.delete(name=cache.name)
        except Exception:  # noqa: BLE001
            pass
    return out


def main() -> None:
    client = genai.Client(api_key=config.GOOGLE_API_KEY)
    results = []
    for model in MODELS:
        r = trial(client, model)
        results.append(r)
        print(r, flush=True)
    # Capability check only: the same prompt with the schema repeated once, so it passes the 1,024 token minimum.
    # This padded prompt is synthetic. It is NOT what the SQL agent sends.
    padded = SYSTEM_PROMPT + "\n\n(Schema repeated for the caching trial only.)\n" + SCHEMA_DESCRIPTION
    r = trial(client, config.GEMINI_MODEL, padded, "padded prompt (synthetic: schema repeated once)")
    results.append(r)
    print(r, flush=True)
    rows = []
    for r in results:
        if r["created"]:
            outcome = f"cache created. Request used {r.get('cached_tokens')} cached of {r.get('input_tokens')} input tokens" + (f". Error on use: {r['error']}" if r.get("error") else "")
        else:
            outcome = f"cache NOT created: {r['error']}"
        rows.append(f"| `{r['model']}` | {r['label']} | {r['prompt_tokens']} | {outcome} |")
    update_section("phase5_explicit_cache", f"""### Explicit context caching trial (SQL system prompt)

Command: `python evals/explicit_cache_trial.py`. The cached content is the SQL system prompt (schema, rules and 3 few-shot examples).

| Model | Prompt | Prompt tokens | Result |
|---|---|---|---|
{chr(10).join(rows)}
""")


if __name__ == "__main__":
    main()
