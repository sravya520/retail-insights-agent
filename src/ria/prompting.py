"""Prompt layout. The long, stable part of a prompt (instructions, schema, few-shot examples) should come first
and the changing part (question, retrieved context) last, so that a repeated prefix can be cached by the API.

  stable_first    (production)  [system: stable text] [human: variable text]
  question_first  (baseline for the caching experiment)  [human: variable text, then the stable text]

Only the caching experiment uses question_first. See evals/caching_experiment.py.
"""
from __future__ import annotations

from ria import config


def build_messages(stable: str, variable: str) -> list[tuple[str, str]]:
    if config.PROMPT_LAYOUT == "question_first":
        return [("human", f"{variable}\n\n{stable}")]
    return [("system", stable), ("human", variable)]
