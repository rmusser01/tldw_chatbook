"""Research budget ledger (task-16323, extends ADR-068's engine contract).

Port of the reserve-and-settle pattern (mole's enforced budgets +
tldw_server limits.py's structured research_limit_exceeded errors) onto the
engine's units of work: searches (query fan-out), fetched docs (raw results
processed), runtime (wall-clock), and token reservations (infrastructure
only until the pipeline reports usage).
"""

import time

import pytest

from tldw_chatbook.Research_Interop.research_budget import (
    BudgetLedger,
    ResearchLimitExceeded,
)


def test_missing_limits_mean_unlimited():
    ledger = BudgetLedger.from_limits({})

    assert ledger.remaining_searches() is None
    assert ledger.remaining_docs() is None
    ledger.reserve_searches(10_000)
    ledger.settle_searches(10_000)
    assert ledger.allot_docs(10_000) == 10_000
    ledger.settle_docs(10_000)
    ledger.check_runtime()  # no deadline configured: never raises


def test_searches_reserve_then_settle_with_structured_error():
    ledger = BudgetLedger.from_limits({"max_searches": 3})

    ledger.reserve_searches(2)
    ledger.settle_searches(2)
    ledger.reserve_searches(1)  # exactly at the cap: allowed

    with pytest.raises(ResearchLimitExceeded) as excinfo:
        ledger.reserve_searches(1)
    assert excinfo.value.limit_key == "max_searches"
    assert str(excinfo.value).startswith("research_limit_exceeded:max_searches")


def test_settle_above_reservation_records_overshoot_never_negative():
    ledger = BudgetLedger.from_limits({"max_searches": 5})

    ledger.reserve_searches(2)
    ledger.settle_searches(3)  # measured overshoot: settle actual usage

    snap = ledger.snapshot()
    assert snap["searches_used"] == 3
    assert snap["searches_overshoot"] == 1


def test_allot_docs_caps_at_remaining_budget():
    ledger = BudgetLedger.from_limits({"max_fetched_docs": 2})

    assert ledger.allot_docs(5) == 2
    ledger.settle_docs(2)

    with pytest.raises(ResearchLimitExceeded) as excinfo:
        ledger.allot_docs(1)
    assert excinfo.value.limit_key == "max_fetched_docs"


def test_runtime_deadline_raises_structured_error():
    ledger = BudgetLedger.from_limits({"max_runtime_seconds": 0.0})

    with pytest.raises(ResearchLimitExceeded) as excinfo:
        ledger.check_runtime()
    assert excinfo.value.limit_key == "max_runtime_seconds"


def test_runtime_none_deadline_never_raises():
    ledger = BudgetLedger.from_limits({})
    time.sleep(0.01)
    ledger.check_runtime()


def test_token_reservations_capped_and_settled():
    ledger = BudgetLedger.from_limits({"max_tokens": 1000})

    ledger.reserve_tokens(600)
    ledger.reserve_tokens(400)  # exactly at the cap: allowed
    with pytest.raises(ResearchLimitExceeded) as excinfo:
        ledger.reserve_tokens(1)
    assert excinfo.value.limit_key == "max_tokens"

    ledger.settle_tokens(750)  # actual usage below reservation
    snap = ledger.snapshot()
    assert snap["tokens_reserved"] == 1000
    assert snap["tokens_settled"] == 750


def test_snapshot_is_persistable_and_reports_all_axes():
    ledger = BudgetLedger.from_limits(
        {"max_searches": 4, "max_fetched_docs": 10, "max_runtime_seconds": 60.0}
    )
    ledger.reserve_searches(1)
    ledger.settle_searches(2)
    ledger.settle_docs(3)
    ledger.settle_tokens(5)  # estimate-settled: flag must be True here

    snap = ledger.snapshot()

    assert snap == {
        "limits": {
            "max_searches": 4,
            "max_fetched_docs": 10,
            "max_runtime_seconds": 60.0,
            "max_tokens": None,
        },
        "searches_used": 2,
        "searches_overshoot": 1,
        "docs_used": 3,
        "tokens_reserved": 0,
        "tokens_settled": 5,
        "tokens_estimated": True,
        "runtime_elapsed_s": snap["runtime_elapsed_s"],  # float, asserted below
    }
    assert isinstance(snap["runtime_elapsed_s"], float)
    assert snap["runtime_elapsed_s"] >= 0.0


def test_invalid_limit_values_fall_back_to_unlimited():
    ledger = BudgetLedger.from_limits(
        {"max_searches": "many", "max_fetched_docs": None, "max_runtime_seconds": -5}
    )
    assert ledger.remaining_searches() is None
    assert ledger.remaining_docs() is None
    ledger.check_runtime()


# --- token enforcement (task-16329) ----------------------------------------------

def test_check_tokens_raises_once_settled_usage_reaches_budget():
    ledger = BudgetLedger.from_limits({"max_tokens": 100})

    ledger.settle_tokens(99)
    ledger.check_tokens()  # still under

    ledger.settle_tokens(1)
    with pytest.raises(ResearchLimitExceeded) as excinfo:
        ledger.check_tokens()
    assert excinfo.value.limit_key == "max_tokens"


def test_check_tokens_no_budget_never_raises():
    ledger = BudgetLedger.from_limits({})
    ledger.settle_tokens(10**9)
    ledger.check_runtime()
    ledger.check_tokens()


def test_snapshot_marks_tokens_as_estimates():
    ledger = BudgetLedger.from_limits({"max_tokens": 50})
    ledger.settle_tokens(20)

    assert ledger.snapshot()["tokens_estimated"] is True
    assert ledger.snapshot()["tokens_settled"] == 20


# --- Qodo remediation (task-16814) ------------------------------------------------

def test_allot_docs_zero_batch_returns_zero_even_with_exhausted_budget():
    ledger = BudgetLedger.from_limits({"max_fetched_docs": 0})

    assert ledger.allot_docs(0) == 0  # nothing to process: no budget failure


def test_tokens_estimated_reflects_exactness_of_settled_usage():
    ledger = BudgetLedger.from_limits({"max_tokens": 1000})
    ledger.settle_tokens(10, exact=True)
    assert ledger.snapshot()["tokens_estimated"] is False

    ledger.settle_tokens(10)  # estimate-settled
    assert ledger.snapshot()["tokens_estimated"] is True


def test_release_searches_returns_unused_reservations():
    ledger = BudgetLedger.from_limits({"max_searches": 3})

    ledger.reserve_searches(3)
    ledger.release_searches(2)  # fan-out stopped early: 2 never executed
    ledger.settle_searches(1)

    assert ledger.remaining_searches() == 2


# --- resume restores spend rather than re-granting it (task-18060) -------------


def test_from_snapshot_restores_spend_rather_than_regranting_it():
    """execute_run rebuilt the ledger from limits on every entry and never read
    budget_ledger.json back, so a resumed run was granted its full budget
    again. With resume routine rather than exceptional, that is a budget leak."""
    from tldw_chatbook.Research_Interop.research_budget import BudgetLedger

    original = BudgetLedger.from_limits({"max_searches": 10})
    original.reserve_searches(4)
    original.settle_searches(4)
    snapshot = original.snapshot()

    restored = BudgetLedger.from_snapshot(snapshot, {"max_searches": 10})

    assert restored.remaining_searches() == 6


def test_from_snapshot_without_a_snapshot_is_a_fresh_ledger():
    """A run that has never executed has no snapshot; it must start whole."""
    from tldw_chatbook.Research_Interop.research_budget import BudgetLedger

    restored = BudgetLedger.from_snapshot(None, {"max_searches": 10})

    assert restored.remaining_searches() == 10


def test_from_snapshot_prefers_current_limits_over_stale_snapshot_limits():
    """A plan-review checkpoint can patch a run's limits AFTER it has
    already executed once and written a snapshot under the OLD limits. The
    current value must win -- a snapshot's stale limits overriding a fresh
    patch is the same bug class the Qodo PR 1766 fix addressed for
    max_iterations in local_research_engine.py."""
    from tldw_chatbook.Research_Interop.research_budget import BudgetLedger

    original = BudgetLedger.from_limits({"max_searches": 3})
    snapshot = original.snapshot()  # snapshot carries the STALE limit (3)

    restored = BudgetLedger.from_snapshot(snapshot, {"max_searches": 20})

    assert restored.max_searches == 20
    assert restored.remaining_searches() == 20


def test_from_snapshot_restores_docs_and_tokens_too():
    """Searches are not the only spendable budget; a partial restore would
    silently re-grant whichever counter it forgot."""
    from tldw_chatbook.Research_Interop.research_budget import BudgetLedger

    original = BudgetLedger.from_limits(
        {"max_searches": 10, "max_fetched_docs": 8, "max_tokens": 1000}
    )
    original.settle_docs(3)
    original.settle_tokens(250, exact=True)
    snapshot = original.snapshot()

    restored = BudgetLedger.from_snapshot(snapshot, None)

    assert restored.docs_used == 3
    assert restored.tokens_settled == 250
    assert restored.max_fetched_docs == 8
