"""Compare an agent result to a gold result for execution accuracy.

Rules:
  * Same number of rows.
  * Every gold cell must match a different cell in the agent row. Extra agent columns are allowed
    (for example a product name next to a revenue figure).
  * Numbers match within a relative tolerance of 0.5%. Strings match ignoring case and outer spaces.
  * If `ordered` is True the rows must be in the same order. Otherwise row order does not matter.
"""
from __future__ import annotations

import math
from decimal import Decimal
from typing import Any, Sequence

NUMERIC_TOLERANCE = 0.005


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float, Decimal)) and not isinstance(v, bool)


def cell_match(gold: Any, agent: Any, tol: float = NUMERIC_TOLERANCE) -> bool:
    if gold is None or agent is None:
        return gold is None and agent is None
    if _is_number(gold) and _is_number(agent):
        return math.isclose(float(gold), float(agent), rel_tol=tol, abs_tol=1e-9)
    return str(gold).strip().lower() == str(agent).strip().lower()


def row_match(gold_row: Sequence[Any], agent_row: Sequence[Any], tol: float = NUMERIC_TOLERANCE) -> bool:
    """Each gold cell must be matched by a distinct agent cell."""
    def assign(i: int, used: frozenset[int]) -> bool:
        if i == len(gold_row):
            return True
        return any(j not in used and cell_match(gold_row[i], agent_row[j], tol) and assign(i + 1, used | {j})
                   for j in range(len(agent_row)))
    return len(agent_row) >= len(gold_row) and assign(0, frozenset())


def results_match(gold_rows: Sequence[Sequence[Any]], agent_rows: Sequence[Sequence[Any]], ordered: bool = False,
                  tol: float = NUMERIC_TOLERANCE) -> bool:
    if len(gold_rows) != len(agent_rows):
        return False
    if ordered:
        return all(row_match(g, a, tol) for g, a in zip(gold_rows, agent_rows))
    remaining = list(agent_rows)
    for g in gold_rows:
        for i, a in enumerate(remaining):
            if row_match(g, a, tol):
                del remaining[i]
                break
        else:
            return False
    return True
