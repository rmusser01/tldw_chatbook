"""Elapsed budget never resets across generations or clock anomalies."""

from dataclasses import replace

import pytest

from Tests.DB.test_automatic_wake_attempts import claim, survivor
from Tests.DB.test_automatic_work_budget import (
    _automatic_work_db,  # noqa: F401
    chain,
    reserve,
)
from tldw_chatbook.Agents.automatic_work_budget import (
    AutomaticWorkLimits,
    AutomaticWorkRefused,
)


@pytest.fixture
def clock(db):
    from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger

    values = [1000.0, 0.0]
    db.automatic_work = AutomaticWorkLedger(
        db, wall_clock=lambda: values[0], monotonic_clock=lambda: values[1]
    )
    return values


def test_deadline_starts_at_first_acceptance_and_never_renews(db, clock):
    chain_id = chain(db, wall_seconds=10)
    reserve(db, chain_id, "first")
    clock[:] = [2000.0, 1000.0]
    assert db.automatic_work.snapshot(chain_id).started_at is None
    assert db.automatic_work.commit("first", owner_id="owner")
    clock[:] = [2005.0, 1005.0]
    reserve(db, chain_id, "second")
    db.automatic_work.commit("second", owner_id="owner")
    assert db.automatic_work.snapshot(chain_id).deadline_at == 2010.0
    clock[:] = [2010.0, 1010.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        reserve(db, chain_id, "third")
    db.close()
    assert db.automatic_work.snapshot(chain_id).pause_reason == "wall_budget"


def test_reservation_cannot_be_committed_after_deadline(db, clock):
    chain_id = chain(db, wall_seconds=5)
    reserve(db, chain_id, "first")
    db.automatic_work.commit("first", owner_id="owner")
    reserve(db, chain_id, "later")
    clock[:] = [1006.0, 6.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        db.automatic_work.commit("later", owner_id="owner")
    assert db.automatic_work.snapshot(chain_id).used["generation"] == 1


def test_clock_reversal_is_durable_review_state(db, clock):
    chain_id = chain(db)
    reserve(db, chain_id, "first")
    db.automatic_work.commit("first", owner_id="owner")
    clock[:] = [999.0, 1.0]
    with pytest.raises(AutomaticWorkRefused, match="clock_reversed"):
        reserve(db, chain_id, "second")
    db.close()
    clock[:] = [1010.0, 10.0]
    with pytest.raises(AutomaticWorkRefused, match="clock_reversed"):
        reserve(db, chain_id, "second")
    assert db.automatic_work.snapshot(chain_id).status == "review_required"


def test_monotonic_elapsed_prevents_slow_wall_clock_from_extending_budget(db, clock):
    chain_id = chain(db, wall_seconds=10)
    reserve(db, chain_id, "first")
    db.automatic_work.commit("first", owner_id="owner")
    clock[:] = [1001.0, 11.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        reserve(db, chain_id, "second")


@pytest.mark.parametrize("observe_expiry_first", [False, True])
def test_expired_monotonic_budget_stays_expired_on_new_handle(
    db, clock, observe_expiry_first
):
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB
    from tldw_chatbook.DB.automatic_work import AutomaticWorkLedger

    chain_id = chain(db, wall_seconds=10)
    reserve(db, chain_id, "first")
    db.automatic_work.commit("first", owner_id="owner")
    clock[:] = [1001.0, 11.0]
    if observe_expiry_first:
        with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
            reserve(db, chain_id, "second")
    with db.connection() as conn:
        database_path = conn.execute("PRAGMA database_list").fetchone()[2]
    reopened = AgentRunsDB(database_path)
    try:
        reopened.automatic_work = AutomaticWorkLedger(
            reopened, wall_clock=lambda: clock[0], monotonic_clock=lambda: clock[1]
        )
        with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
            reserve(reopened, chain_id, "second")
        assert reopened.automatic_work.snapshot(chain_id).used["generation"] == 1
    finally:
        reopened.close()


def test_live_limits_can_reduce_but_not_replenish_chain(db, clock):
    chain_id = chain(db, generations=1)
    reserve(db, chain_id, "first")
    with pytest.raises(AutomaticWorkRefused, match="generation_budget"):
        db.automatic_work.commit(
            "first",
            owner_id="owner",
            limits=replace(AutomaticWorkLimits(), generations=0),
        )
    assert db.automatic_work.commit("first", owner_id="owner")
    with pytest.raises(AutomaticWorkRefused, match="generation_budget"):
        db.automatic_work.reserve(
            chain_id,
            reservation_id="second",
            owner_id="owner",
            kind="generation",
            amount=1,
            limits=replace(AutomaticWorkLimits(), generations=99),
        )


@pytest.mark.parametrize("first_admission_refused", [False, True])
def test_recovered_chain_uses_new_process_clock_without_renewing_time(
    db, clock, monkeypatch, first_admission_refused
):
    from tldw_chatbook.DB import automatic_work
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    chain_id = chain(db, wall_seconds=10, budget_tokens=100)
    reserve(db, chain_id, "first")
    db.automatic_work.commit("first", owner_id="owner")
    db.close()
    monkeypatch.setattr(automatic_work, "_CLOCK_OWNER_ID", "replacement-process")
    reopened = AgentRunsDB(db.db_path_str)
    try:
        reopened.automatic_work = automatic_work.AutomaticWorkLedger(
            reopened, wall_clock=lambda: clock[0], monotonic_clock=lambda: clock[1]
        )
        clock[:] = [1005.0, 100.0]
        assert (
            reopened.automatic_work.recover(current_owner_id="replacement-owner") == 0
        )
        if first_admission_refused:
            with pytest.raises(AutomaticWorkRefused, match="tokens_budget"):
                reserve(
                    reopened,
                    chain_id,
                    "second",
                    kind="tokens",
                    amount=101,
                    owner="replacement-owner",
                )
        else:
            reserve(reopened, chain_id, "second", owner="replacement-owner")
            assert reopened.automatic_work.commit(
                "second", owner_id="replacement-owner"
            )
        clock[:] = [1006.0, 111.0]
        with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
            reserve(
                reopened,
                chain_id,
                "child",
                kind="child_launch",
                owner="replacement-owner",
            )
        assert reopened.automatic_work.snapshot(chain_id).deadline_at == 1010.0
    finally:
        reopened.close()


def test_wake_acceptance_rechecks_deadline(db, clock):
    chain_id = chain(db, wall_seconds=5)
    first = survivor(db, chain_id)
    claim(db, chain_id, [first])
    assert db.automatic_work.accept_wake("attempt", owner_id="owner")
    db.automatic_work.complete_wake("attempt", owner_id="owner")
    second = survivor(db, chain_id)
    claim(db, chain_id, [second], attempt="second")
    clock[:] = [1006.0, 6.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        db.automatic_work.accept_wake("second", owner_id="owner")
    assert db.automatic_work.abort_wake("second", owner_id="owner")


@pytest.mark.parametrize("backward", [False, True])
def test_members_share_original_deadline_and_durable_clock_refusal(db, clock, backward):
    from Tests.DB.test_automatic_chat_starts import prepare, source_run

    root, source = source_run(db, wall_seconds=10)
    first = prepare(db, source, "a", "first")
    db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    clock[:] = [1005.0, 5.0]
    second = prepare(db, source, "b", "second")
    db.automatic_work.accept_chat_start(second.id, owner_id="owner")
    assert db.automatic_work.snapshot(second.chain_id).deadline_at == 1010.0
    clock[:] = [1004.0, 6.0] if backward else [1009.0, 11.0]
    reason = "clock_reversed" if backward else "wall_budget"
    with pytest.raises(AutomaticWorkRefused, match=reason):
        reserve(db, second.chain_id, "third")
    db.close()
    clock[:] = [1012.0, 12.0]
    for member in (root, first.chain_id, second.chain_id):
        snapshot = db.automatic_work.snapshot(member)
        assert snapshot.deadline_at == 1010.0
        assert snapshot.pause_reason == reason
        with pytest.raises(AutomaticWorkRefused, match=reason):
            db.automatic_work.check_active(member, owner_id="owner")


def test_refused_descendant_prepare_retains_observed_root_time(db, clock):
    from Tests.DB.test_automatic_chat_starts import prepare, source_run

    root, source = source_run(db, generations=1)
    first = prepare(db, source, "a", "first")
    db.automatic_work.accept_chat_start(first.id, owner_id="owner")
    clock[:] = [1005.0, 5.0]
    with pytest.raises(AutomaticWorkRefused, match="generation_budget"):
        prepare(db, source, "b", "second")
    clock[:] = [1004.0, 6.0]
    with pytest.raises(AutomaticWorkRefused, match="clock_reversed"):
        reserve(db, first.chain_id, "third")
    assert db.automatic_work.snapshot(root).status == "review_required"


# -- TASK-33621.153: time held at the Ask compaction trigger -----------------
#
# Owner decision 2026-11-12: time a chain waits at a compaction_ask hold does
# not count against its automatic-work wall clock. The limit measures only the
# time the work runs, so a /compact long after the pause resumes normally.


def _started(db, *, wall_seconds=10, **limits):
    """A chain whose clock started at wall 1000 / monotonic 0 with one turn."""
    chain_id = chain(db, wall_seconds=wall_seconds, **limits)
    reserve(db, chain_id, "first")
    assert db.automatic_work.commit("first", owner_id="owner")
    return chain_id


def _call(db, chain_id, reservation_id, *, owner="owner"):
    return reserve(db, chain_id, reservation_id, kind="model_call", owner=owner)


def test_time_held_at_the_ask_trigger_does_not_count(db, clock):
    chain_id = _started(db)
    clock[:] = [1004.0, 4.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    clock[:] = [1100.0, 100.0]  # held 96 s, far past the 10 s limit

    _call(db, chain_id, "after-the-hold")

    snapshot = db.automatic_work.snapshot(chain_id)
    assert (snapshot.status, snapshot.pause_reason) == ("active", None)
    assert snapshot.deadline_at == 1106.0, "6 s left, as at the pause"
    clock[:] = [1105.5, 105.5]
    _call(db, chain_id, "still-running")
    clock[:] = [1106.0, 106.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        _call(db, chain_id, "past-the-running-time")


@pytest.mark.parametrize("reason", ["autowake_disabled", "conversation_wake_active"])
def test_other_pauses_keep_the_clock_running(db, clock, reason):
    chain_id = _started(db)
    clock[:] = [1004.0, 4.0]
    db.automatic_work.pause(chain_id, reason)
    clock[:] = [1100.0, 100.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        _call(db, chain_id, "after-the-pause")


def test_an_ask_hold_survives_a_restart(db, clock, monkeypatch):
    from tldw_chatbook.DB import automatic_work
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    chain_id = _started(db)
    clock[:] = [1004.0, 4.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    db.close()
    monkeypatch.setattr(automatic_work, "_CLOCK_OWNER_ID", "replacement-process")
    reopened = AgentRunsDB(db.db_path_str)
    try:
        reopened.automatic_work = automatic_work.AutomaticWorkLedger(
            reopened, wall_clock=lambda: clock[0], monotonic_clock=lambda: clock[1]
        )
        # A new process: its monotonic clock reads lower than the old anchor.
        clock[:] = [1100.0, 2.0]
        owner = "replacement-owner"
        assert reopened.automatic_work.recover(current_owner_id=owner) == 0
        held = reopened.automatic_work.snapshot(chain_id)
        assert (held.status, held.pause_reason) == ("paused", "compaction_ask")

        _call(reopened, chain_id, "after-restart", owner=owner)

        assert reopened.automatic_work.snapshot(chain_id).deadline_at == 1106.0
        clock[:] = [1105.0, 7.0]
        _call(reopened, chain_id, "still-running", owner=owner)
        clock[:] = [1106.0, 8.0]
        with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
            _call(reopened, chain_id, "past-the-running-time", owner=owner)
    finally:
        reopened.close()


def test_a_sleep_during_the_hold_neither_reverses_nor_spends_the_clock(db, clock):
    chain_id = _started(db)
    clock[:] = [1004.0, 4.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    # The machine slept for an hour: the wall clock moved, the monotonic one
    # barely did. Neither reading may refuse the resume or count the hold.
    clock[:] = [4604.0, 10.0]

    _call(db, chain_id, "after-the-sleep")

    assert db.automatic_work.snapshot(chain_id).deadline_at == 4610.0
    # Running time is still bounded by the monotonic clock: 6 s ran here.
    clock[:] = [4605.0, 16.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        _call(db, chain_id, "monotonic-past-the-running-time")


def test_monotonic_time_run_before_the_hold_still_counts(db, clock):
    chain_id = _started(db)
    # The wall clock ran slow: 1 s by the wall, 5 s by the monotonic clock.
    clock[:] = [1001.0, 5.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    clock[:] = [1100.0, 104.0]

    _call(db, chain_id, "after-the-hold")

    clock[:] = [1104.9, 108.9]
    _call(db, chain_id, "still-running")
    clock[:] = [1105.0, 109.0]  # 5 s before the hold + 5 s after it
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        _call(db, chain_id, "past-the-running-time")


def test_a_second_ask_pause_does_not_count_the_first_hold_as_running(db, clock):
    chain_id = _started(db)
    clock[:] = [1004.0, 4.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    clock[:] = [1050.0, 50.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    clock[:] = [1100.0, 100.0]

    _call(db, chain_id, "after-the-hold")

    assert db.automatic_work.snapshot(chain_id).deadline_at == 1106.0


def test_a_refused_first_retry_keeps_the_held_time_out(db, clock):
    chain_id = _started(db, budget_tokens=100)
    clock[:] = [1004.0, 4.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    clock[:] = [1100.0, 100.0]
    with pytest.raises(AutomaticWorkRefused, match="tokens_budget"):
        reserve(db, chain_id, "too-many-tokens", kind="tokens", amount=101)

    snapshot = db.automatic_work.snapshot(chain_id)
    assert (snapshot.status, snapshot.pause_reason) == ("paused", "tokens_budget")
    assert snapshot.deadline_at == 1106.0, "the hold ended once, and stays out"
    clock[:] = [1105.0, 105.0]
    _call(db, chain_id, "still-running")
    clock[:] = [1106.0, 106.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        _call(db, chain_id, "past-the-running-time")


def test_a_pause_that_replaces_the_ask_hold_ends_it(db, clock):
    chain_id = _started(db, wall_seconds=100)
    clock[:] = [1004.0, 4.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    clock[:] = [1050.0, 50.0]
    db.automatic_work.pause(chain_id, "autowake_disabled")

    # 46 s held at the Ask trigger stay out; from here the clock runs again.
    assert db.automatic_work.snapshot(chain_id).deadline_at == 1146.0
    clock[:] = [1140.0, 140.0]
    _call(db, chain_id, "still-running")
    clock[:] = [1146.0, 146.0]
    with pytest.raises(AutomaticWorkRefused, match="wall_budget"):
        _call(db, chain_id, "past-the-running-time")


def test_a_hold_before_the_first_wake_starts_no_clock(db, clock):
    chain_id = chain(db, wall_seconds=10)
    clock[:] = [1004.0, 4.0]
    db.automatic_work.pause(chain_id, "compaction_ask")
    clock[:] = [1100.0, 100.0]
    reserve(db, chain_id, "first")
    assert db.automatic_work.snapshot(chain_id).started_at is None
    assert db.automatic_work.commit("first", owner_id="owner")
    assert db.automatic_work.snapshot(chain_id).deadline_at == 1110.0
