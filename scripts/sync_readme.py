"""Copy result tables from docs/results.md into README.md, so the README never has a retyped number.

README markers look like  <!-- BEGIN sync:name -->  ...  <!-- END sync:name -->.
Run: python scripts/sync_readme.py        (offline)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results.md"
README = ROOT / "README.md"

# marker name -> (results section, first line of the table)
TABLES = {
    "eval_v1_gates": ("eval_v1", "| Gate | Result | Detail |"),
    "eval_v1_obs": ("eval_v1", "| Metric | Held-out | Dev |"),
    "tokens_route": ("phase5_tokens", "| Route | Requests |"),
    "tokens_share": ("phase5_tokens", "| Part of the prompt |"),
    "caching_main": ("phase5_caching", "| Layout | Queries |"),
    "caching_probe": ("phase5_caching", "| Call | Input tokens | Cached tokens |"),
    "explicit_cache": ("phase5_explicit_cache", "| Model | Prompt | Prompt tokens | Result |"),
    "rewrite": ("phase5_rewrite_experiment", "| Variant | Recall@5 | MRR@5 |"),
}


def section(text: str, name: str) -> str:
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    return text.split(begin, 1)[1].split(end, 1)[0]


def table(block: str, first_line: str) -> str:
    lines = block.splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(first_line))
    rows = []
    for l in lines[start:]:
        if not l.startswith("|"):
            break
        rows.append(l)
    return "\n".join(rows)


def main() -> None:
    results = RESULTS.read_text(encoding="utf-8").replace("\r\n", "\n")
    readme = README.read_text(encoding="utf-8").replace("\r\n", "\n")
    for marker, (sec, first) in TABLES.items():
        begin, end = f"<!-- BEGIN sync:{marker} -->", f"<!-- END sync:{marker} -->"
        if begin not in readme:
            print(f"marker {marker} not in README, skipped")
            continue
        body = table(section(results, sec), first)
        head, rest = readme.split(begin, 1)
        _, tail = rest.split(end, 1)
        readme = f"{head}{begin}\n{body}\n{end}{tail}"
    README.write_text(readme, encoding="utf-8")
    left = re.findall(r"BEGIN sync:(\w+)", readme)
    print("synced:", ", ".join(left))


if __name__ == "__main__":
    sys.exit(main())
