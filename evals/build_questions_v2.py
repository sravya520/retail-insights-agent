"""Build evals/questions_v2.jsonl: a fresh held-out set (v2), written BEFORE the dev fixes and committed before any run.

These 20 questions were written from the policy documents and the database schema only. None of them was run
through the system or used to tune a prompt, a retrieval setting or a guardrail. Gold answers for data questions
come from running my gold SQL on the database (no model involved).

v2 is meant to be harder and more varied than v1: casual phrasing, a derived number, a subquery, questions the policy
documents do not cover (expect_not_covered), and boundary out of scope questions.

Run: python evals/build_questions_v2.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from ria.sql_db import connect_readonly  # noqa: E402
from evals.build_questions import TRAJECTORY, cite, DP, LP, SD, WB_DOC, WQ  # noqa: E402

V2 = [
    # data
    dict(id="v01", route="data", ordered=False,
         question="In which month of 2011 were the most orders placed, and how many orders was that?",
         gold_sql="SELECT invoice_month, COUNT(DISTINCT invoice_no) AS orders FROM sales WHERE invoice_year = 2011 GROUP BY invoice_month ORDER BY orders DESC LIMIT 1"),
    dict(id="v02", route="data", ordered=False, question="What was the total revenue from Spain in 2010?",
         gold_sql="SELECT SUM(revenue) AS revenue FROM sales WHERE country = 'Spain' AND invoice_year = 2010"),
    dict(id="v03", route="data", ordered=False,
         question="How many different merchandise products, excluding postage and fees, were sold in November 2011?",
         gold_sql="SELECT COUNT(DISTINCT stock_code) AS products FROM sales WHERE is_merchandise AND invoice_year = 2011 AND invoice_month = 11"),
    dict(id="v04", route="data", ordered=False, question="Which customer returned the most units in 2011, and how many units was that?",
         gold_sql="SELECT customer_id, SUM(quantity) AS units FROM returns WHERE invoice_year = 2011 AND customer_id IS NOT NULL GROUP BY customer_id ORDER BY units DESC, customer_id LIMIT 1"),
    dict(id="v05", route="data", ordered=False, question="What was the revenue in the second half of 2010, from July to December?",
         gold_sql="SELECT SUM(revenue) AS revenue FROM sales WHERE invoice_year = 2010 AND invoice_month BETWEEN 7 AND 12"),
    dict(id="v06", route="data", ordered=False, question="How many orders worth more than 500 pounds were placed in 2011?",
         gold_sql="SELECT COUNT(*) AS big_orders FROM (SELECT invoice_no FROM sales WHERE invoice_year = 2011 GROUP BY invoice_no HAVING SUM(revenue) > 500)"),
    # policy
    dict(id="v07", route="policy", question="I am a wholesale customer and want to cancel a 1,500 pound order before it ships. Is there a fee?",
         gold_citations=[cite(WB_DOC, "5.3")], gold_text="Yes. A cancellation fee of 10 percent of the order value applies to orders of 1,000 pounds or more."),
    dict(id="v08", route="policy", question="What happens to my loyalty points if I do not buy anything for a year?",
         gold_citations=[cite(LP, "5.1")], gold_text="All points expire after 12 months with no earning or redeeming activity."),
    dict(id="v09", route="policy", question="Can you post a can of room spray to Canada?",
         gold_citations=[cite(SD, "4.2")], gold_text="No. Aerosols (and liquids and fragrance oil candles) are not shipped outside the UK and EU."),
    dict(id="v10", route="policy", question="How long do you keep recordings of customer service calls?",
         gold_citations=[cite(DP, "2.4")], gold_text="90 days."),
    dict(id="v11", route="policy", expect_not_covered=True, gold_citations=[],
         question="What are the opening hours of the Northwind Gifts head office?",
         gold_text="Not covered by the policy documents."),
    dict(id="v12", route="policy", expect_not_covered=True, gold_citations=[],
         question="Can I pay for an order in instalments?", gold_text="Not covered by the policy documents."),
    # both
    dict(id="v13", route="both", ordered=False,
         question="How many orders were placed in November 2010, and what is the cut-off time for same-day dispatch?",
         gold_sql="SELECT COUNT(DISTINCT invoice_no) AS orders FROM sales WHERE invoice_year = 2010 AND invoice_month = 11",
         gold_citations=[cite(SD, "1.1")], gold_text="Orders placed before 2pm on a business day are dispatched the same day."),
    dict(id="v14", route="both", ordered=False,
         question="Which country outside the United Kingdom had the most orders in 2011, and what does standard EU delivery cost?",
         gold_sql="SELECT country, COUNT(DISTINCT invoice_no) AS orders FROM sales WHERE country <> 'United Kingdom' AND invoice_year = 2011 GROUP BY country ORDER BY orders DESC, country LIMIT 1",
         gold_citations=[cite(SD, "3.1")], gold_text="EU standard delivery costs 9.95 pounds (free from 120 pounds)."),
    dict(id="v15", route="both", ordered=False,
         question="What was the revenue in December 2010, and how long is the warranty on electrical items?",
         gold_sql="SELECT SUM(revenue) AS revenue FROM sales WHERE invoice_year = 2010 AND invoice_month = 12",
         gold_citations=[cite(WQ, "2.2")], gold_text="24 months from the delivery date."),
    dict(id="v16", route="both", ordered=False,
         question="How many different customers bought something in 2011, and what discount does a wholesale order of 2,500 pounds get?",
         gold_sql="SELECT COUNT(DISTINCT customer_id) AS customers FROM sales WHERE invoice_year = 2011 AND customer_id IS NOT NULL",
         gold_citations=[cite(WB_DOC, "3.1")], gold_text="8 percent (orders of 2,000 to 4,999 pounds)."),
    # out of scope
    dict(id="v17", route="out_of_scope", question="Translate good morning into French."),
    dict(id="v18", route="out_of_scope", question="What is the best way to wrap a fragile gift so it does not break?"),
    dict(id="v19", route="out_of_scope", question="Give me the personal email address of the chief executive."),
    dict(id="v20", route="out_of_scope", question="Which films are showing at the cinema tonight?"),
]


def main() -> None:
    con = connect_readonly()
    out = []
    for item in V2:
        item = dict(item)
        route = item.pop("route")
        item.update(split="heldout", expected_route=route, expected_trajectory=TRAJECTORY[route])
        if item.get("gold_sql"):
            cursor = con.execute(item["gold_sql"])
            item["gold_columns"] = [d[0] for d in cursor.description]
            item["gold_answer"] = [[v if v is None or isinstance(v, (int, float, str, bool)) else str(v) for v in r]
                                   for r in cursor.fetchall()]
        out.append(item)
    path = ROOT / "evals" / "questions_v2.jsonl"
    path.write_text("\n".join(json.dumps(i, ensure_ascii=False) for i in out) + "\n", encoding="utf-8")
    counts = {r: sum(i["expected_route"] == r for i in out) for r in TRAJECTORY}
    print(len(out), counts)
    for i in out:
        if i.get("gold_answer"):
            print(i["id"], i["gold_answer"])


if __name__ == "__main__":
    main()
