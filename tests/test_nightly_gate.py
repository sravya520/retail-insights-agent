"""The nightly job must fail when a gate fails. This runs evals/run_evals.py in --report mode (no API calls)
on hand-made records, once with a clean set and once with an out of scope question that was answered."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data" / "db" / "sales.duckdb"
pytestmark = pytest.mark.skipif(not DB.exists(), reason="needs data/db/sales.duckdb (python -I -m ria.sql_db)")

REFUSAL = ("I can only help with questions about Northwind Gifts sales data and its policy documents "
           "(returns, warranty, shipping, wholesale terms, loyalty programme, data privacy). "
           "I cannot help with that question.")


def run_report(run_id: str, records: list[dict]) -> subprocess.CompletedProcess:
    run_dir = ROOT / "evals" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        (run_dir / "records.jsonl").write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
        return subprocess.run([sys.executable, "evals/run_evals.py", "--report", run_id, "--no-write", "--label", "gate_test"],
                              cwd=ROOT, capture_output=True, text=True, timeout=120)
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


def oos_record(answer: str, trajectory: list[str], route: str) -> dict:
    return {"id": "h17", "split": "heldout", "question": "Write a Python function.", "expected_route": "out_of_scope",
            "expected_trajectory": ["router", "refuse"], "route": route, "trajectory": trajectory, "answer": answer,
            "verified": None, "removed": [], "error": None, "latency_s": 1.0}


def test_clean_records_exit_zero():
    res = run_report("gate-test-clean", [oos_record(REFUSAL, ["router", "refuse"], "out_of_scope")])
    assert res.returncode == 0, res.stdout[-800:] + res.stderr[-800:]


def test_answered_out_of_scope_question_fails_the_gate_with_exit_one():
    bad = oos_record("def reverse(s): return s[::-1]", ["router", "rag_agent", "combiner", "verifier"], "policy")
    res = run_report("gate-test-bad", [bad])
    assert res.returncode == 1, res.stdout[-800:] + res.stderr[-800:]
    assert "FAIL" in res.stdout
