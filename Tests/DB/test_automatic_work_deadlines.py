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
