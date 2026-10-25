from ria.verifier import FALLBACK_ANSWER, extract_numbers, is_supported, support_values, verify_answer

ROWS = [("RABBIT NIGHT LIGHT", 34556.71999999993), ("PAPER CHAIN KIT 50'S CHRISTMAS", 28985.04)]
CHUNKS = [
    {"doc": "Returns and Refunds", "clause_id": "7.4", "text": "7.4 Damaged wholesale goods. Report within 48 hours of delivery."},
    {"doc": "Returns and Refunds", "clause_id": "2.1", "text": "2.1 Retail return window. Within 30 days of the delivery date."},
]


def values(text):
    return [round(n.value, 6) for n in extract_numbers(text)]


def test_extract_numbers_formats():
    assert values("Revenue was £34,556.72 and 15.4% of 2010") == [34556.72, 15.4, 2010]
    assert values("About £1.2m or 3k units") == [1200000, 3000]
    assert values("Stock 85123A and B2B and the 5th day") == []  # codes and ordinals are not numbers
    assert values("Delivery takes 5-10 business days") == [5, 10]
    assert values("15 percent restocking fee") == [15]


def test_rounding_tolerance():
    supports = support_values(ROWS)

    def n(s):
        return extract_numbers(s)[0]

    assert is_supported(n("£34,556.72"), supports)
    assert is_supported(n("£34,557"), supports)       # rounded to whole pounds
    assert is_supported(n("£34.6k"), supports)        # rounded to hundreds
    assert not is_supported(n("£34,556"), supports)   # truncated, not rounded
    assert not is_supported(n("£35,000"), supports)
    assert is_supported(extract_numbers("15.4%")[0], [0.154])  # a ratio written as a percentage


def test_clean_answer_is_kept_unchanged():
    ans = "The top product was RABBIT NIGHT LIGHT at £34,556.72. Second was PAPER CHAIN KIT 50'S CHRISTMAS at £28,985.04."
    res = verify_answer(ans, "top products in November 2011", sql_rows=ROWS)
    assert res.verified and not res.removed and res.answer == ans


def test_verifier_removes_fabricated_number_and_logs_it():
    ans = "The top product was RABBIT NIGHT LIGHT at £34,556.72. Revenue grew 87% on last year."
    res = verify_answer(ans, "top products in November 2011", sql_rows=ROWS)
    assert not res.verified
    assert res.answer == "The top product was RABBIT NIGHT LIGHT at £34,556.72."
    assert [(r.reason, r.detail) for r in res.removed] == [("unsupported_number", "87")]
    assert res.to_log()["removed"][0]["sentence"] == "Revenue grew 87% on last year."


def test_verifier_removes_fake_citation():
    ans = ("Wholesale damage must be reported within 48 hours [Returns and Refunds §7.4, p.2]. "
           "Wholesale buyers get 90 days [Returns and Refunds §9.9, p.3].")
    res = verify_answer(ans, "wholesale damaged item", chunks=CHUNKS)
    assert res.answer == "Wholesale damage must be reported within 48 hours [Returns and Refunds §7.4, p.2]."
    assert res.removed[0].reason == "invalid_citation" and "§9.9" in res.removed[0].detail


def test_number_must_match_the_cited_chunk_not_just_any_chunk():
    ans = "Retail customers have 48 days [Returns and Refunds §2.1, p.1]."  # 48 is in chunk 7.4, not in 2.1
    res = verify_answer(ans, "return window", chunks=CHUNKS)
    assert res.fallback_used and res.answer == FALLBACK_ANSWER
    assert res.removed[0].reason == "unsupported_number"


def test_policy_number_in_cited_chunk_is_supported_and_citation_after_full_stop_kept():
    ans = "Retail customers have 30 days. [Returns and Refunds §2.1, p.1]"
    res = verify_answer(ans, "return window", chunks=CHUNKS)
    assert res.verified and res.answer == ans


def test_numbers_in_the_question_are_allowed():
    res = verify_answer("After 40 days the claim may be refused [Returns and Refunds §7.4, p.2].",
                        "Can a wholesale customer return a damaged item after 40 days?", chunks=CHUNKS)
    assert res.verified


def test_list_markers_and_row_count_are_not_fabrications():
    ans = "1. RABBIT NIGHT LIGHT at £34,556.72\n2. PAPER CHAIN KIT 50'S CHRISTMAS at £28,985.04\nThese are the 2 products found."
    res = verify_answer(ans, "top products", sql_rows=ROWS)
    assert res.verified, res.removed
    assert res.answer == ans  # list markers are kept


def test_number_with_no_data_at_all_is_removed():
    res = verify_answer("Revenue was £5,000.", "revenue?", sql_rows=[])
    assert res.fallback_used and res.removed[0].detail == "5000"


def test_verifier_output_is_always_clean():
    ans = "Fine sentence. Made up 12345 units. Another fake [Returns and Refunds §1.1, p.1]. Okay again."
    first = verify_answer(ans, "q", chunks=CHUNKS)
    second = verify_answer(first.answer, "q", chunks=CHUNKS)
    assert first.removed and not second.removed and second.answer == first.answer
