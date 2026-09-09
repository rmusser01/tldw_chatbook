"""Pure, conservative accounting for the frozen source-reader experiment matrix."""

import math
from collections import defaultdict
from statistics import median

ARMS = ("direct", "retrieval", "reader")
MAX_ROWS = 10_000
MAX_TEXT_LENGTH = 256
MAX_INTEGER = 2**53 - 1
MAX_COST_RATIO = 0.80
MAX_LATENCY_DELTA_SECONDS = 10.0


def _nonnegative_integer(value: object) -> bool:
    return type(value) is int and 0 <= value <= MAX_INTEGER


def _number(value: object) -> float | None:
    if type(value) not in (int, float):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) and number >= 0 else None


def _key(row: object) -> tuple[str, int, str] | None:
    if not isinstance(row, dict):
        return None
    case, repeat, arm = row.get("case_id"), row.get("repeat"), row.get("arm")
    if (
        not isinstance(case, str)
        or not case.strip()
        or len(case) > MAX_TEXT_LENGTH
        or not _nonnegative_integer(repeat)
        or not isinstance(arm, str)
        or arm not in ARMS
    ):
        return None
    return case, repeat, arm


def _bounded_rows(rows: object, name: str, reasons: set[str]) -> list:
    if not isinstance(rows, list):
        reasons.add(f"invalid_{name}")
        return []
    if len(rows) > MAX_ROWS:
        reasons.add(f"too_many_{name}")
        return rows[:MAX_ROWS]
    return rows


def _index(rows: list, name: str, reasons: set[str]) -> dict:
    indexed = {}
    for row in rows:
        key = _key(row)
        if key is None:
            reasons.add(f"invalid_{name}_identity")
        elif key in indexed:
            reasons.add(f"duplicate_{name}")
            indexed[key] = None
        else:
            indexed[key] = row
    # No arbitrary winner when multiple rows claim one matrix position.
    return {key: row for key, row in indexed.items() if row is not None}


def _valid_grade(row: dict) -> bool:
    correct, total = row.get("essential_correct"), row.get("essential_total")
    return (
        type(row.get("success")) is bool
        and type(row.get("critical_error")) is bool
        and _nonnegative_integer(correct)
        and _nonnegative_integer(total)
        and correct <= total
    )


def _sum_costs(values: list[float], reasons: set[str]) -> float | None:
    try:
        return math.fsum(values)
    except OverflowError:
        reasons.add("cost_total_overflow")
        return None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    # Scaling avoids overflow in statistics.median's even-sized middle sum.
    return median([value / 2 for value in values]) * 2


def _paired_metrics(groups: list, attempts: dict, arm: str, baseline: str) -> dict:
    pairs = []
    for case, repeat in groups:
        candidate = attempts.get((case, repeat, arm))
        reference = attempts.get((case, repeat, baseline))
        ratio = delta = None
        if candidate is not None and reference is not None:
            cost = _number(candidate.get("cost_usd"))
            base_cost = _number(reference.get("cost_usd"))
            if cost is not None and base_cost is not None and base_cost > 0:
                value = cost / base_cost
                ratio = value if math.isfinite(value) else None
            latency = _number(candidate.get("latency_seconds"))
            base_latency = _number(reference.get("latency_seconds"))
            if latency is not None and base_latency is not None:
                delta = latency - base_latency
        pairs.append(
            {
                "case_id": case,
                "repeat": repeat,
                "cost_ratio": ratio,
                "latency_delta_seconds": delta,
            }
        )
    ratios = [pair["cost_ratio"] for pair in pairs if pair["cost_ratio"] is not None]
    deltas = [
        pair["latency_delta_seconds"]
        for pair in pairs
        if pair["latency_delta_seconds"] is not None
    ]
    return {
        "count": len(pairs),
        "cost_pair_count": len(ratios),
        "latency_pair_count": len(deltas),
        "median_cost_ratio": _median(ratios) if len(ratios) == len(pairs) else None,
        "median_latency_delta_seconds": (
            _median(deltas) if len(deltas) == len(pairs) else None
        ),
        "per_case": pairs,
    }


def _quality(expected: dict, attempts: dict, grades: dict, arm: str) -> dict:
    keys = [key for key in expected if key[2] == arm]
    scored = [key for key in keys if key in grades]
    recalls = defaultdict(list)
    for key in scored:
        grade = grades[key]
        total = grade["essential_total"]
        recalls[key[0]].append(grade["essential_correct"] / total if total else 1.0)
    complete = (
        bool(keys) and len(scored) == len(keys) and all(key in attempts for key in keys)
    )
    successes = sum(
        grades[key]["success"] and attempts.get(key, {}).get("status") == "ok"
        for key in scored
    )
    return {
        "graded_count": len(scored),
        "expected_count": len(keys),
        "task_success_rate": successes / len(keys) if complete else None,
        "macro_essential_recall": (
            math.fsum(math.fsum(values) / len(values) for values in recalls.values())
            / len(recalls)
            if complete
            else None
        ),
        "critical_error_count": sum(grades[key]["critical_error"] for key in scored),
    }


def _quality_matches(quality: dict, grades: dict, groups: list, arm: str) -> bool:
    candidate, direct = quality[arm], quality["direct"]
    for metric in ("task_success_rate", "macro_essential_recall"):
        if candidate[metric] is None or direct[metric] is None:
            return False
        if candidate[metric] < direct[metric]:
            return False
    return not _new_critical_errors(grades, groups, arm)


def _new_critical_errors(grades: dict, groups: list, arm: str) -> bool:
    return any(
        grades.get((case, repeat, arm), {}).get("critical_error", False)
        and not grades.get((case, repeat, "direct"), {}).get("critical_error", False)
        for case, repeat in groups
    )


def summarize_comparison(
    attempts: list[dict], grades: list[dict], expected: list[dict]
) -> dict:
    """Summarize all observed spend and gate a paired three-arm experiment.

    Args:
        attempts: One attempt per frozen key, with status, nullable total USD
            cost (including every worker/main/embedding/reranker request), and
            wall-clock latency. Only status ``ok`` counts as execution success.
            Optional ``known_cost_usd`` preserves billed call subtotals when
            total cost is unknown; it never makes incomplete spend complete.
        grades: Human grades keyed by case, repeat and arm. Empty essential-fact
            checklists have recall 1.0; answer success is a separate grade.
        expected: Host-owned frozen ``case_id``, ``repeat``, ``arm`` rows. Each
            case/repeat requires all three arms, even if an arm never ran.

    Returns:
        A JSON-safe report and bounded reason codes. Incomplete/invalid evidence
        takes precedence over rejection; known failures still appear in reasons.
        Macro recall weights cases equally after averaging their repetitions.
        Rows are capped at 10,000 per input; oversized inputs are inconclusive
        and their spend is explicitly incomplete. Identity strings are bounded
        to 256 codepoints and integers to JSON's exactly representable range.
        Extra metadata is ignored and is never copied into the report.
    """
    uncertain: set[str] = set()
    rejected: set[str] = set()
    observed_count = len(attempts) if isinstance(attempts, list) else 0
    expected_count = len(expected) if isinstance(expected, list) else 0
    attempt_rows = _bounded_rows(attempts, "attempts", uncertain)
    grade_rows = _bounded_rows(grades, "grades", uncertain)
    expected_rows = _bounded_rows(expected, "expected", uncertain)
    expected_index = _index(expected_rows, "expected", uncertain)
    attempt_index = _index(attempt_rows, "attempts", uncertain)
    grade_index = _index(grade_rows, "grades", uncertain)
    groups = sorted({key[:2] for key in expected_index})
    if not expected_index:
        uncertain.add("empty_expected_matrix")
    if any(
        (case, repeat, arm) not in expected_index
        for case, repeat in groups
        for arm in ARMS
    ):
        uncertain.add("incomplete_expected_matrix")
    for name, indexed in (("attempts", attempt_index), ("grades", grade_index)):
        if expected_index.keys() - indexed.keys():
            uncertain.add(f"missing_{name}")
        if indexed.keys() - expected_index.keys():
            uncertain.add(f"extra_{name}")

    valid_grades = {}
    for key, grade in grade_index.items():
        if _valid_grade(grade):
            valid_grades[key] = grade
        else:
            uncertain.add("invalid_grades")
    fact_totals = defaultdict(set)
    for key, grade in valid_grades.items():
        if key in expected_index:
            fact_totals[key[0]].add(grade["essential_total"])
    if any(len(totals) > 1 for totals in fact_totals.values()):
        uncertain.add("inconsistent_essential_totals")
    failed_grade_keys = {
        _key(grade)
        for grade in grade_rows
        if _key(grade) is not None and _valid_grade(grade) and not grade["success"]
    }

    costs: dict[str, list[float]] = {arm: [] for arm in (*ARMS, "unassigned")}
    unknown = dict.fromkeys(costs, 0)
    failures = dict.fromkeys(costs, 0)
    for attempt in attempt_rows:
        if not isinstance(attempt, dict):
            unknown["unassigned"] += 1
            continue
        arm = attempt.get("arm")
        arm = arm if isinstance(arm, str) and arm in ARMS else "unassigned"
        cost = _number(attempt.get("cost_usd"))
        known_cost = _number(attempt.get("known_cost_usd"))
        if "known_cost_usd" in attempt and known_cost is None:
            uncertain.add("invalid_known_cost")
        if cost is None:
            unknown[arm] += 1
            uncertain.add("unknown_or_invalid_cost")
            if known_cost is not None:
                costs[arm].append(known_cost)
        else:
            costs[arm].append(cost)
        if _number(attempt.get("latency_seconds")) is None:
            uncertain.add("invalid_latency")
        status = attempt.get("status")
        if not isinstance(status, str) or not status or len(status) > MAX_TEXT_LENGTH:
            uncertain.add("invalid_status")
        if status != "ok" or _key(attempt) in failed_grade_keys:
            failures[arm] += 1

    by_arm = {
        arm: {
            "known_total_usd": _sum_costs(costs[arm], uncertain),
            "unknown_cost_count": unknown[arm],
        }
        for arm in costs
    }
    known_total = _sum_costs(
        [cost for values in costs.values() for cost in values], uncertain
    )
    observed_costs_complete = (
        not sum(unknown.values())
        and isinstance(attempts, list)
        and len(attempt_rows) == observed_count
        and known_total is not None
    )
    matrix_complete = (
        bool(expected_index)
        and len(expected_index)
        == expected_count
        == len(attempt_index)
        == observed_count
        and expected_index.keys() == attempt_index.keys()
        and not any("expected" in reason for reason in uncertain)
    )
    quality = {
        arm: _quality(expected_index, attempt_index, valid_grades, arm) for arm in ARMS
    }
    reader_pairs = _paired_metrics(groups, attempt_index, "reader", "direct")
    retrieval_pairs = _paired_metrics(groups, attempt_index, "retrieval", "reader")
    if reader_pairs["median_cost_ratio"] is None:
        uncertain.add("undefined_cost_ratio")
    if reader_pairs["median_latency_delta_seconds"] is None:
        uncertain.add("undefined_latency_delta")
    if sum(failures.values()):
        rejected.add("failed_attempts")
    if all(
        quality[arm]["task_success_rate"] is not None for arm in ARMS
    ) and not _quality_matches(quality, valid_grades, groups, "reader"):
        rejected.add("reader_quality_regression")
    if _new_critical_errors(valid_grades, groups, "reader"):
        rejected.add("reader_new_critical_errors")
    ratio = reader_pairs["median_cost_ratio"]
    delta = reader_pairs["median_latency_delta_seconds"]
    if ratio is not None and ratio > MAX_COST_RATIO:
        rejected.add("cost_gate_failed")
    if delta is not None and delta > MAX_LATENCY_DELTA_SECONDS:
        rejected.add("latency_gate_failed")

    reasons = uncertain | rejected
    if uncertain:
        decision = "inconclusive"
    elif rejected:
        decision = "reject"
    elif (
        _quality_matches(quality, valid_grades, groups, "retrieval")
        and by_arm["retrieval"]["known_total_usd"]
        <= by_arm["reader"]["known_total_usd"]
    ):
        decision = "prefer_retrieval"
        reasons.add("retrieval_matches_quality_at_lower_or_equal_cost")
    else:
        decision = "pilot_candidate"
        reasons.add("all_pilot_gates_passed")
    return {
        "decision": decision,
        "reasons": sorted(reasons),
        "expected_attempt_count": expected_count,
        "observed_attempt_count": observed_count,
        "spend": {
            "known_total_usd": known_total,
            "complete": observed_costs_complete and matrix_complete,
            "observed_costs_complete": observed_costs_complete,
            "unknown_cost_count": sum(unknown.values()),
            "by_arm": by_arm,
        },
        "failures": {"total": sum(failures.values()), "by_arm": failures},
        "quality": quality,
        "paired": {"reader_direct": reader_pairs, "retrieval_reader": retrieval_pairs},
    }
