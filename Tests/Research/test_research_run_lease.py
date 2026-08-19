"""Run leases (task-18060): exactly one executor may hold a run."""

import re
from datetime import datetime, timezone

import pytest

from tldw_chatbook.Research_Interop.local_research_service import (
    LeaseBudgetExhausted,
    LocalResearchService,
)

# _now()/_timestamp_after() must always render this exact shape: fixed
# microsecond precision (never omitted) and a trailing "Z". See
# _format_timestamp's docstring for why the precision can't be optional.
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{6}Z$")


def _service() -> LocalResearchService:
    return LocalResearchService(":memory:")


def test_first_claim_succeeds_and_returns_a_lease_id():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    lease_id = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)

    assert isinstance(lease_id, str) and lease_id


def test_second_claim_is_refused_while_the_lease_is_live():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)

    assert service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60) is None


def test_zero_lease_seconds_produces_an_already_expired_lease():
    """A zero lease_seconds must yield a lease a subsequent claim can take
    over immediately -- deterministically, with no time.sleep(). What's
    under test is the string comparison in claim_run's UPDATE guard, not
    elapsed wall-clock time, so no real waiting is needed or wanted.
    """
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    first = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)
    assert isinstance(first, str) and first

    second = service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)
    assert isinstance(second, str) and second
    assert second != first


def test_negative_lease_seconds_produces_an_already_expired_lease():
    """A negative lease_seconds clamps to "expires now" rather than being
    treated literally (which would try to lease *before* it was granted);
    a subsequent claim must still be able to take over immediately.
    """
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    first = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=-5)
    assert isinstance(first, str) and first

    second = service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)
    assert isinstance(second, str) and second


def test_now_and_timestamp_after_produce_mutually_comparable_strings():
    """Regression guard for task-18060 finding 1: whatever ``_now()`` and
    ``_timestamp_after()`` produce must be directly comparable as plain
    strings, since claim_run's atomicity is a single string comparison in
    a SQL WHERE clause, not a parsed-datetime comparison.
    """
    service = _service()
    now = service._now()
    zero_offset = service._timestamp_after(0)
    later = service._timestamp_after(60)

    assert _TIMESTAMP_RE.match(now)
    assert _TIMESTAMP_RE.match(zero_offset)
    assert _TIMESTAMP_RE.match(later)
    assert now <= zero_offset <= later


def test_format_timestamp_is_stable_at_a_whole_second_boundary():
    """Pins the exact bug the reviewer found: plain ``datetime.isoformat()``
    drops the fractional-seconds field when microsecond == 0, and ``"."``
    (0x2E) sorts below ``"Z"`` (0x5A). Before the fix, a whole-second
    timestamp with no fractional part would sort *below* an earlier
    timestamp that still had trailing digits, silently reordering leases.
    ``_format_timestamp`` must pin microsecond precision so this can't
    recur regardless of how ``_now``/``_timestamp_after`` are called.
    """
    whole_second = datetime(2026, 8, 18, 8, 31, 1, 0, tzinfo=timezone.utc)
    fractional = datetime(2026, 8, 18, 8, 31, 1, 123456, tzinfo=timezone.utc)

    whole_str = LocalResearchService._format_timestamp(whole_second)
    fractional_str = LocalResearchService._format_timestamp(fractional)

    assert whole_str == "2026-08-18T08:31:01.000000Z"
    assert _TIMESTAMP_RE.match(whole_str)
    assert _TIMESTAMP_RE.match(fractional_str)
    # Chronological order must equal string order.
    assert whole_str < fractional_str


# NOTE: a lease of 0 seconds expires the instant it is granted, so these tests
# exercise takeover deterministically. Do NOT use time.sleep() to age a lease:
# a wall-clock dependency makes the suite flaky on a loaded machine, and the
# behaviour under test is the comparison against `leased_until`, not duration.


def test_a_stale_lease_can_be_taken_over():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)

    assert service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)


def test_a_displaced_worker_cannot_renew_or_release():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    stale = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)
    service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)

    assert service.renew_lease(run["id"], lease_id=stale, lease_seconds=60) is False
    assert service.release_lease(run["id"], lease_id=stale) is False
    assert service.holds_lease(run["id"], lease_id=stale) is False


def test_reclaim_stops_at_the_retry_budget():
    """Review finding 1: once the retry budget is spent, claim_run raises
    LeaseBudgetExhausted rather than returning None -- overloading None to
    mean both "someone else holds it live" and "the budget is spent" left
    callers unable to tell the two apart, and the caller MUST respond to
    them differently (see the engine-level tests in
    test_local_research_engine.py for the two different responses)."""
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    for _ in range(3):
        assert service.claim_run(
            run["id"], worker_id="w", lease_seconds=0, max_attempts=3
        )

    with pytest.raises(LeaseBudgetExhausted) as excinfo:
        service.claim_run(run["id"], worker_id="w", lease_seconds=0, max_attempts=3)
    assert excinfo.value.run_id == run["id"]
    assert excinfo.value.attempts == 3
    assert excinfo.value.max_attempts == 3


def test_release_frees_the_run_for_the_next_executor():
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    lease = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)

    assert service.release_lease(run["id"], lease_id=lease) is True
    assert service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)


def test_a_clean_release_resets_the_retry_budget():
    """Regression guard for task-18060 finding 1: a healthy claim->release
    cycle must NOT burn the crash-retry budget. Without resetting
    lease_attempts on release, several healthy cycles would exhaust
    max_attempts on their own, and the genuine crash recovery below would
    be refused even though the run was never actually abandoned more than
    once.
    """
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    # Several healthy claim -> release cycles by well-behaved executors.
    for _ in range(3):
        lease = service.claim_run(
            run["id"], worker_id="w", lease_seconds=60, max_attempts=3
        )
        assert lease is not None
        assert service.release_lease(run["id"], lease_id=lease) is True

    # A genuine crash: claimed but never released, lease left to expire.
    abandoned = service.claim_run(
        run["id"], worker_id="w", lease_seconds=0, max_attempts=3
    )
    assert abandoned is not None

    # The rescue claim must still succeed -- the prior clean releases must
    # not have consumed the budget that this single abandonment is spending.
    rescue = service.claim_run(
        run["id"], worker_id="rescuer", lease_seconds=60, max_attempts=3
    )
    assert rescue is not None


def test_a_live_lease_at_the_retry_budget_declines_instead_of_raising():
    """task-3 report finding 1: ``claim_run`` treated ANY previous lease --
    live or expired -- as a "reclaim" for budget-check purposes, keyed only
    on ``previous is not None``. So once ``lease_attempts`` reached
    ``max_attempts``, a second executor merely racing a perfectly healthy,
    still-live lease got ``LeaseBudgetExhausted`` instead of a routine
    decline. The engine's caller responds to that exception by calling
    ``fail_run`` (see ``local_research_engine.py`` around the
    ``LeaseBudgetExhausted`` handler), so a run whose executor was actively
    working got failed out from under it with a false "claimed and
    abandoned N time(s)" message.

    Reproduces the reviewer's repro: two abandonments (expired leases, never
    released) spend two of the three attempts, then a THIRD, healthy claim
    takes the run and its lease is still live. A concurrent claim against
    that live lease must be declined (return None, raise nothing) -- the
    budget check only applies to RECLAIMING an EXPIRED lease.
    """
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    # Two abandonments: claimed with an already-expired lease, never released.
    for _ in range(2):
        assert service.claim_run(
            run["id"], worker_id="dead-executor", lease_seconds=0, max_attempts=3
        )

    # A healthy third claim: attempt 3 of 3, but its lease is LIVE, not
    # abandoned.
    holder_lease = service.claim_run(
        run["id"], worker_id="healthy-executor", lease_seconds=60, max_attempts=3
    )
    assert holder_lease is not None

    # A racing second executor must be declined, not told the budget is
    # exhausted.
    racer_result = service.claim_run(
        run["id"], worker_id="racer", lease_seconds=60, max_attempts=3
    )
    assert racer_result is None

    # The run itself must be untouched: not failed, and still held by its
    # legitimate, healthy holder.
    run_row = service.get_run(run["id"])
    assert run_row["status"] != "failed"
    assert service.holds_lease(run["id"], lease_id=holder_lease) is True
