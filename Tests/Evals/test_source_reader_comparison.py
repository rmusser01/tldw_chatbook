"""Decision gates must preserve the frozen comparison matrix and every cost."""

import json

import pytest

from tldw_chatbook.Evals.source_reader.comparison import (
    summarize_comparison as summarize,
)


def matrix(cases=("a",), repeats=1):
    expected = [
        {"case_id": case, "repeat": repeat, "arm": arm}
        for case in cases
        for repeat in range(repeats)
        for arm in ("direct", "retrieval", "reader")
    ]
    attempts = [
        dict(
            key,
            status="ok",
            cost_usd={"direct": 1.0, "retrieval": 0.9, "reader": 0.5}[key["arm"]],
            latency_seconds={"direct": 4.0, "retrieval": 3.0, "reader": 6.0}[
                key["arm"]
            ],
        )
        for key in expected
    ]
    grades = [
        dict(
            key,
            success=True,
            critical_error=False,
            essential_correct=2,
            essential_total=2,
        )
        for key in expected
    ]
    return attempts, grades, expected


def row(rows, arm, case="a", repeat=0):
    return next(
        item
        for item in rows
        if item["arm"] == arm and item["case_id"] == case and item["repeat"] == repeat
    )


def test_pilot_uses_paired_metrics_and_includes_every_arm_spend():
    report = summarize(*matrix())

    assert report["decision"] == "pilot_candidate"
    assert report["spend"]["known_total_usd"] == pytest.approx(2.4)
    assert report["spend"]["complete"] is True
    assert report["failures"]["total"] == 0
    paired = report["paired"]["reader_direct"]
    assert paired["median_cost_ratio"] == 0.5
    assert paired["median_latency_delta_seconds"] == 2.0
    assert paired["per_case"] == [
        {"case_id": "a", "repeat": 0, "cost_ratio": 0.5, "latency_delta_seconds": 2.0}
    ]
    assert report["quality"]["reader"]["macro_essential_recall"] == 1.0
    assert json.loads(json.dumps(report, allow_nan=False)) == report


def test_unpaired_medians_cannot_manufacture_savings():
    attempts, grades, expected = matrix(("a", "b", "c"))
    for case, direct, reader in (("a", 1, 2), ("b", 100, 50), ("c", 101, 90)):
        row(attempts, "direct", case)["cost_usd"] = direct
        row(attempts, "reader", case)["cost_usd"] = reader
        row(attempts, "retrieval", case)["cost_usd"] = 200
    report = summarize(attempts[::-1], grades[::-1], expected)

    # Ratio of unpaired medians is 50/100=0.5; paired median is 90/101.
    assert report["paired"]["reader_direct"]["median_cost_ratio"] == pytest.approx(
        90 / 101
    )
    assert report["decision"] == "reject"
    assert "cost_gate_failed" in report["reasons"]


def test_latency_gate_uses_matched_cases_and_repetitions():
    attempts, grades, expected = matrix(repeats=3)
    for repeat, direct, reader in ((0, 0, 20), (1, 100, 0), (2, 101, 121)):
        row(attempts, "direct", repeat=repeat)["latency_seconds"] = direct
        row(attempts, "reader", repeat=repeat)["latency_seconds"] = reader
    report = summarize(attempts, grades, expected)

    assert report["paired"]["reader_direct"]["median_latency_delta_seconds"] == 20
    assert report["decision"] == "reject"
    assert "latency_gate_failed" in report["reasons"]


@pytest.mark.parametrize("collection", ["attempts", "grades", "expected"])
@pytest.mark.parametrize("change", ["missing", "duplicate", "extra"])
def test_incomplete_or_ambiguous_matrix_is_inconclusive(collection, change):
    attempts, grades, expected = matrix()
    rows = {"attempts": attempts, "grades": grades, "expected": expected}[collection]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows.append(dict(rows[-1]))
    else:
        rows.append(dict(rows[-1], case_id="unexpected"))
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert report["reasons"]


def test_an_entire_missing_case_is_not_inferred_out_of_expected_matrix():
    attempts, grades, expected = matrix(("a", "b"))
    report = summarize(attempts[:3], grades[:3], expected)

    assert report["decision"] == "inconclusive"
    assert report["expected_attempt_count"] == 6
    assert report["spend"]["complete"] is False


@pytest.mark.parametrize("field", ["cost_usd", "latency_seconds"])
@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), -1, True, "0.1"])
def test_missing_or_invalid_usage_cannot_pass(field, value):
    attempts, grades, expected = matrix()
    row(attempts, "reader")[field] = value
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert report["reasons"]
    json.dumps(report, allow_nan=False)
    if field == "cost_usd":
        assert report["spend"]["known_total_usd"] == 1.9
        assert report["spend"]["complete"] is False
        assert report["spend"]["unknown_cost_count"] == 1


def test_zero_direct_cost_has_no_financial_ratio():
    attempts, grades, expected = matrix()
    row(attempts, "direct")["cost_usd"] = 0.0
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert report["paired"]["reader_direct"]["median_cost_ratio"] is None
    assert "undefined_cost_ratio" in report["reasons"]


@pytest.mark.parametrize("arm", ["direct", "retrieval", "reader"])
def test_cheap_failed_attempts_never_count_as_savings(arm):
    attempts, grades, expected = matrix()
    row(attempts, arm).update(status="provider_error", cost_usd=0.01)
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "reject"
    assert report["failures"]["total"] == 1
    assert report["failures"]["by_arm"][arm] == 1
    assert "failed_attempts" in report["reasons"]
    assert report["spend"]["known_total_usd"] == pytest.approx(
        {"direct": 1.41, "retrieval": 1.51, "reader": 1.91}[arm]
    )


def test_unsuccessful_human_grade_blocks_savings_even_with_ok_transport():
    attempts, grades, expected = matrix()
    row(grades, "reader")["success"] = False
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "reject"
    assert report["quality"]["reader"]["task_success_rate"] == 0.0
    assert report["failures"]["total"] == 1


def test_omitted_human_review_is_inconclusive():
    attempts, _, expected = matrix()
    report = summarize(attempts, [], expected)

    assert report["decision"] == "inconclusive"
    assert report["quality"]["reader"]["task_success_rate"] is None
    assert "missing_grades" in report["reasons"]


def test_macro_recall_prevents_large_easy_case_from_hiding_lost_small_case():
    attempts, grades, expected = matrix(("a", "b"))
    for arm in ("direct", "retrieval", "reader"):
        row(grades, arm, "a").update(essential_correct=0, essential_total=1)
        row(grades, arm, "b").update(essential_correct=100, essential_total=100)
    row(grades, "direct", "a")["essential_correct"] = 1
    row(grades, "direct", "b")["essential_correct"] = 50
    report = summarize(attempts, grades, expected)

    assert report["quality"]["direct"]["macro_essential_recall"] == 0.75
    assert report["quality"]["reader"]["macro_essential_recall"] == 0.5
    assert report["decision"] == "reject"
    assert "reader_quality_regression" in report["reasons"]


def test_new_critical_error_is_not_cancelled_by_improvement_on_another_case():
    attempts, grades, expected = matrix(("a", "b"))
    row(grades, "direct", "a")["critical_error"] = True
    row(grades, "reader", "b")["critical_error"] = True
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "reject"
    assert "reader_new_critical_errors" in report["reasons"]


@pytest.mark.parametrize("cost", [0.4, 0.5])
def test_adequate_retrieval_at_equal_or_lower_cost_is_preferred(cost):
    attempts, grades, expected = matrix()
    row(attempts, "retrieval")["cost_usd"] = cost
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "prefer_retrieval"
    assert "retrieval_matches_quality_at_lower_or_equal_cost" in report["reasons"]


def test_cheaper_retrieval_does_not_win_when_essential_recall_is_lower():
    attempts, grades, expected = matrix()
    row(attempts, "retrieval")["cost_usd"] = 0.1
    row(grades, "retrieval")["essential_correct"] = 1
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "pilot_candidate"


@pytest.mark.parametrize(
    "field,value",
    [
        ("success", 1),
        ("critical_error", "false"),
        ("essential_correct", -1),
        ("essential_correct", 3),
        ("essential_total", float("nan")),
        ("essential_total", True),
        ("essential_correct", 1.0),
    ],
)
def test_invalid_human_grades_are_inconclusive(field, value):
    attempts, grades, expected = matrix()
    row(grades, "reader")[field] = value
    assert summarize(attempts, grades, expected)["decision"] == "inconclusive"


def test_empty_essential_checklist_remains_defined_for_no_answer_cases():
    attempts, grades, expected = matrix()
    for grade in grades:
        grade.update(essential_correct=0, essential_total=0)
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "pilot_candidate"
    assert report["quality"]["reader"]["macro_essential_recall"] == 1.0


def test_duplicate_and_extra_attempt_spend_is_still_counted():
    attempts, grades, expected = matrix()
    attempts.append(dict(attempts[-1], cost_usd=2))
    attempts.append(dict(attempts[-1], case_id="unexpected", cost_usd=3))
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert report["spend"]["known_total_usd"] == 7.4
    assert report["spend"]["by_arm"]["reader"]["known_total_usd"] == 5.5


@pytest.mark.parametrize(
    "key,value", [("case_id", ""), ("repeat", -1), ("repeat", True), ("arm", "other")]
)
def test_invalid_attempt_identity_keeps_its_observed_spend(key, value):
    attempts, grades, expected = matrix()
    row(attempts, "reader")[key] = value
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert report["spend"]["known_total_usd"] == 2.4


def test_empty_manifest_cannot_manufacture_a_vacuous_pass():
    assert summarize([], [], [])["decision"] == "inconclusive"


@pytest.mark.parametrize("repeat", [0, 1])
def test_changing_the_frozen_fact_checklist_cannot_inflate_recall(repeat):
    attempts, grades, expected = matrix(repeats=2)
    row(grades, "reader", repeat=repeat).update(essential_correct=1, essential_total=1)
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert "inconsistent_essential_totals" in report["reasons"]


def test_failure_in_duplicate_human_grades_is_still_reported():
    attempts, grades, expected = matrix()
    grades.append(dict(grades[-1], success=False))
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert report["failures"]["total"] == 1
    assert "failed_attempts" in report["reasons"]


def test_unknown_main_usage_retains_the_known_worker_subtotal():
    attempts, grades, expected = matrix()
    row(attempts, "reader").update(
        status="unknown_usage", cost_usd=None, known_cost_usd=0.25
    )
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert report["spend"]["known_total_usd"] == 2.15
    assert report["spend"]["by_arm"]["reader"]["known_total_usd"] == 0.25
    assert report["spend"]["unknown_cost_count"] == 1
    assert report["spend"]["complete"] is False
    assert report["spend"]["observed_costs_complete"] is False
    assert report["paired"]["reader_direct"]["median_cost_ratio"] is None


def test_known_subtotal_is_not_added_to_a_complete_attempt_total():
    attempts, grades, expected = matrix()
    row(attempts, "reader")["known_cost_usd"] = 0.5
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "pilot_candidate"
    assert report["spend"]["known_total_usd"] == 2.4
    assert report["spend"]["by_arm"]["reader"]["known_total_usd"] == 0.5


@pytest.mark.parametrize("cost", [None, 0.5])
@pytest.mark.parametrize("subtotal", [float("nan"), float("inf"), -1, True, "0.25"])
def test_invalid_subtotals_never_enter_known_spend_or_a_pass(cost, subtotal):
    attempts, grades, expected = matrix()
    row(attempts, "reader").update(cost_usd=cost, known_cost_usd=subtotal)
    report = summarize(attempts, grades, expected)

    assert report["decision"] == "inconclusive"
    assert report["spend"]["known_total_usd"] == (1.9 if cost is None else 2.4)
    assert "invalid_known_cost" in report["reasons"]
    json.dumps(report, allow_nan=False)
