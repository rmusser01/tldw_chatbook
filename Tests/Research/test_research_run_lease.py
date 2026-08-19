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


def test_releasing_an_expired_lease_does_not_reset_the_retry_budget():
    """PR-1822 review follow-up: ``release_lease`` matched on ``lease_id``
    only, with no liveness check, so it reset ``lease_attempts`` to 0 for an
    EXPIRED lease too. An executor that stalls past expiry, notices, and
    exits through its finally block therefore "cleanly released" a lease it
    no longer held -- and a systematically stalling-but-alive executor could
    loop claim -> expire -> release forever without ever spending the crash
    budget, defeating the "fail a run whose executor keeps dying" contract
    (AC #1b).

    Only a release while the lease is still LIVE counts as clean. Releasing
    an expired lease must still free the run for the next claimant but must
    keep the abandonment on the books.
    """
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    # Two stalls: claimed, lease allowed to expire, executor wakes and
    # releases what it no longer holds.
    for _ in range(2):
        lease = service.claim_run(
            run["id"], worker_id="stalling", lease_seconds=0, max_attempts=3
        )
        assert lease is not None
        # Zero-second lease: already expired at the moment of release.
        assert service.release_lease(run["id"], lease_id=lease) is True

    # The run is claimable again (the release DID clear the lease)...
    third = service.claim_run(
        run["id"], worker_id="stalling", lease_seconds=0, max_attempts=3
    )
    assert third is not None
    assert service.release_lease(run["id"], lease_id=third) is True

    # ...but the third stall must now exhaust the budget: three
    # abandonments, none of them clean.
    with pytest.raises(LeaseBudgetExhausted):
        service.claim_run(
            run["id"], worker_id="rescuer", lease_seconds=60, max_attempts=3
        )


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


# --- task-3 report (fourth fix-up): terminal-claim race, self-renewing
# expired leases, and lease-conditional terminal writes -------------------


def test_claim_run_refuses_a_cancelled_run():
    """task-3 report finding 1: ``claim_run`` restricted acquisition by
    lease expiry only, never by run status, while ``execute_run``'s own
    terminal check happens BEFORE the claim. A cancellation landing between
    that check and the claim let a terminal run be claimed and executed
    (resurrected) anyway. The status condition now lives in the SAME atomic
    UPDATE as the lease-expiry check, so a cancelled run can never be
    claimed, first claim or not.
    """
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    service.cancel_run(run["id"])

    assert service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60) is None


def test_claim_run_refuses_a_completed_run_even_with_an_expired_lease():
    """A run that finished (and thus never released, but no longer needs
    to be) must not be reclaimable just because its lease looks expired."""
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)

    service.complete_run(run["id"])

    assert service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60) is None


def test_renew_lease_after_expiry_returns_false_even_without_takeover():
    """task-3 report finding 3: ``renew_lease`` matched on ``run_id`` and
    ``lease_id`` only, so a worker whose lease already expired could extend
    it as long as nobody had taken over YET -- a stalled worker resurrecting
    a claim it had already lost, contradicting takeover. The lease must
    still be LIVE for a renewal to land, independent of whether a second
    claimant has shown up.
    """
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    lease = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)
    assert lease is not None

    # No takeover has happened -- the lease is simply expired.
    assert service.renew_lease(run["id"], lease_id=lease, lease_seconds=60) is False
    # holds_lease agrees: an expired lease is not "held" either.
    assert service.holds_lease(run["id"], lease_id=lease) is False


def test_complete_run_with_a_stale_lease_id_is_a_noop():
    """task-3 report finding 4: a standalone ``holds_lease()`` check
    followed by an unconditional write is check-then-act -- a takeover
    between the two still lets the stale write land. ``complete_run``'s own
    UPDATE now matches ``lease_id`` at the SQL level, so a displaced
    executor's completion is a no-op instead of a race.
    """
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    stale = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)
    rescuer = service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)
    assert rescuer is not None

    result = service.complete_run(run["id"], lease_id=stale)

    assert result["status"] != "completed"
    current = service.get_run(run["id"])
    assert current["status"] != "completed"
    assert service.holds_lease(run["id"], lease_id=rescuer) is True


def test_fail_run_with_a_stale_lease_id_is_a_noop():
    """Same contract as complete_run's sibling test, for fail_run."""
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    stale = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=0)
    rescuer = service.claim_run(run["id"], worker_id="worker-b", lease_seconds=60)
    assert rescuer is not None

    result = service.fail_run(run["id"], error_msg="boom", lease_id=stale)

    assert result["status"] != "failed"
    current = service.get_run(run["id"])
    assert current["status"] != "failed"
    assert service.holds_lease(run["id"], lease_id=rescuer) is True


def test_complete_run_with_a_matching_lease_id_still_lands():
    """The lease-conditional UPDATE must not become a no-op for the
    legitimate holder -- only a mismatched (or absent-but-expected) lease
    id blocks the write."""
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")
    lease = service.claim_run(run["id"], worker_id="worker-a", lease_seconds=60)
    assert lease is not None

    result = service.complete_run(run["id"], lease_id=lease)

    assert result["status"] == "completed"


def test_complete_run_without_a_lease_id_is_unconditional_as_before():
    """A caller that never held a lease (the LeaseBudgetExhausted path, or
    any pre-existing caller outside the leased-execution flow) omits
    ``lease_id`` -- the write must remain unconditional, matching behavior
    before task-3's fourth fix-up."""
    service = _service()
    run = service.launch_run(query="q", autonomy_mode="autonomous")

    result = service.complete_run(run["id"])

    assert result["status"] == "completed"
