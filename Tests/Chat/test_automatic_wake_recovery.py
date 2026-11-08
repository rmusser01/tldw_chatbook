"""Saved results and uncertain attempts are discovered without attention badges."""

import pytest

from Tests.Chat.test_automatic_wake_budget import (
    close_rig,
    result_for,
)
from Tests.Chat.test_console_fleet_wake import _controller_rig, _settle
from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def rig(tmp_path):
    return _controller_rig(tmp_path)


def replacement(rig):
    controller = rig[7]
    wake = ConsoleFleetWakeCoordinator(controller)
    controller._fleet_wake = wake
    wake.wire()
    return wake


@pytest.mark.asyncio
async def test_known_unclaimed_result_is_recovered_without_an_unseen_mark(rig):
    _, _, db, _, session, gateway, _, _ = rig
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    child = result_for(rig, chain)
    wake = replacement(rig)
    try:
        await wake.recover()
        assert await _settle(lambda: db.get_run(child)["wake_delivered_at"] is not None)
        assert len(gateway.payloads) == 1
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted", [False, True])
async def test_interrupted_attempt_stays_paused_after_its_badge_was_cleared(
    rig, accepted
):
    _, _, db, _, session, gateway, _, _ = rig
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    child = result_for(rig, chain)
    db.automatic_work.claim_wake(
        chain,
        attempt_id="old",
        owner_id="previous-runtime",
        session_id=session.id,
        run_ids=[child],
    )
    if accepted:
        db.automatic_work.accept_wake("old", owner_id="previous-runtime")
    wake = replacement(rig)
    try:
        await wake.recover()
        assert wake.has_pending(session.id)
        assert wake.pause_reason(session.id) == "interrupted_work"
        snapshot = db.automatic_work.snapshot(chain)
        assert snapshot.status == "review_required"
        assert snapshot.used["generation"] + snapshot.reserved["generation"] == 1
        assert gateway.payloads == []
        assert db.get_run(child)["wake_delivered_at"] is None
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_remount_seeding_does_not_recover_or_invalidate_a_live_owner(rig):
    _, _, db, _, session, _, _, controller = rig
    wake = controller.fleet_wake
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    child = result_for(rig, chain)
    db.automatic_work.claim_wake(
        chain,
        attempt_id="live",
        owner_id=wake._owner_id,
        session_id=session.id,
        run_ids=[child],
    )
    db.automatic_work.accept_wake("live", owner_id=wake._owner_id)
    try:
        wake.seed_from_marks()
        db.close()
        assert (
            db.automatic_work.read_attempt("live", owner_id=wake._owner_id).state
            == "accepted"
        )
        assert db.automatic_work.snapshot(chain).status == "active"
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_failed_startup_audit_is_visible_and_never_admits_work(rig, monkeypatch):
    import sqlite3

    wake = replacement(rig)
    notices = []
    wake.delivery_ui_hook = notices.append

    def fail_audit(**kwargs):
        raise sqlite3.OperationalError("recovery write failed")

    monkeypatch.setattr(rig[2].automatic_work, "recover", fail_audit)
    try:
        await wake.recover()
        assert not await wake.wait_for_recovery()
        assert wake.pause_reason(rig[4].id) == "history_unavailable"
        assert notices == [rig[4].id]
        assert rig[5].payloads == []
    finally:
        await close_rig(rig)


@pytest.mark.asyncio
async def test_a_console_with_no_runs_db_has_nothing_to_recover(rig):
    """TASK-38971: with no agent-runs DB there is no wake to audit.

    The bridge, and the runs DB it carries, is absent when there is no
    durable ChaChaNotes DB. ``recover`` still read
    ``self._runs_db().automatic_work``, raised AttributeError, logged "wake
    recovery failed" and kept goal runs fenced as ``history_unavailable``
    for the runtime's life, since recovery runs once.
    """
    from loguru import logger

    rig[7]._agent_bridge = None
    wake = replacement(rig)
    warnings = []
    sink = logger.add(warnings.append, format="{message}", level="WARNING")
    try:
        await wake.recover()
        assert await wake.wait_for_recovery()
        assert wake.recovery_result is None
        assert wake.pause_reason(rig[4].id) is None
        assert not [line for line in warnings if "wake recovery failed" in line]
    finally:
        logger.remove(sink)
        await close_rig(rig)


@pytest.mark.asyncio
async def test_recovery_failure_explains_results_in_a_later_resumed_session(
    rig, monkeypatch
):
    db, store, session = rig[2], rig[3], rig[4]
    chain = db.automatic_work.create_chain(session.id, root_submission_id="manual")
    result_for(rig, chain)
    store.close_session(session.id)
    assert store.sessions() == []
    wake = replacement(rig)

    def fail_audit(**kwargs):
        raise OSError("history unavailable")

    monkeypatch.setattr(db.automatic_work, "recover", fail_audit)
    try:
        await wake.recover()
        resumed = store.create_session(title="Resumed")
        resumed.persisted_conversation_id = session.id
        assert wake.seed_from_marks() == 1
        assert wake.has_pending(session.id)
        assert wake.pause_reason(session.id) == "history_unavailable"
        assert not await wake.wait_for_recovery()
    finally:
        await close_rig(rig)
