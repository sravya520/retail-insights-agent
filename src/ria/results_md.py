"""Write a named section of docs/results.md. Scripts call this so numbers come from real runs."""
from __future__ import annotations

from pathlib import Path

from ria.config import RESULTS_MD


def update_section(name: str, content: str, path: Path = RESULTS_MD) -> None:
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    block = f"{begin}\n{content.strip()}\n{end}"
    text = path.read_text(encoding="utf-8") if path.exists() else "# Results\n"
    if begin in text and end in text:
        head, rest = text.split(begin, 1)
        _, tail = rest.split(end, 1)
        text = head + block + tail
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
