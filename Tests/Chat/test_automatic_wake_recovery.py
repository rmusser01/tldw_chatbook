"""Saved results and uncertain attempts are discovered without attention badges."""

import threading

import pytest

from Tests.Chat.test_automatic_wake_budget import (
    close_rig,
    result_for,
)
from Tests.Chat.test_console_fleet_wake import (
    _controller_rig,
    _drain,
    _settle,
    _survivor,
    _terminal_subagent_run,
)
from tldw_chatbook.Chat.chat_persistence_service import ChatPersistenceService
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole, ConsoleRunStatus
from tldw_chatbook.Chat.console_fleet_wake import ConsoleFleetWakeCoordinator

pytestmark = pytest.mark.bootstrap_profile


@pytest.fixture
def rig(tmp_path):
    return _controller_rig(tmp_path)


def replacement(rig):
    controller = rig[7]
    wake = ConsoleFleetWakeCoordinator(controller)
    controller._fleet_wake = wake
    wake.wire(app=rig[1])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("move", ["binding", "persistence", "close"])
async def test_a_wake_whose_owner_moves_during_its_reply_save_is_refused(rig, move):
    """TASK-39159: a wake's pre-provider reply save (TASK-39119) adopted the
    saved row only when the chat's binding was unchanged, but went on to the
    provider either way, so the reply streamed into a row whose saved copy was
    not linked to it. The save is held while the chat is rebound (or its
    persistence replaced), as a manual turn's mix-in attempt would be refused.
    Every turn that reaches the provider must stream into an adopted row.
    A plain close in that window (review r13p-rev1 M3) ends the turn as a
    closed session, not as a refusal written into the closed chat."""
    chacha, _, runs_db, store, session, gateway, _, controller = rig
    try:
        persistence = store.persistence
        real_save = persistence.save_pending_assistant
        entered, release = threading.Event(), threading.Event()
        saved: list[str] = []

        def held_save(conversation_id, message_id, *args):
            real_save(conversation_id, message_id, *args)
            saved.append(message_id)
            if len(saved) == 1:
                entered.set()
                release.wait(10)

        persistence.save_pending_assistant = held_save
        adopted_at_entry: list[bool] = []

        def provider_entered():
            reply = next(
                row
                for row in reversed(store.messages_for_session(session.id))
                if row.role is ConsoleMessageRole.ASSISTANT
            )
            adopted_at_entry.append(reply.persisted_message_id is not None)

        gateway.on_stream = provider_entered
        outcomes: list[object] = []
        real_submit = controller.submit_draft

        async def recorded_submit(*args, **kwargs):
            try:
                result = await real_submit(*args, **kwargs)
            except BaseException as exc:
                outcomes.append(exc)
                raise
            outcomes.append(result)
            return result

        controller.submit_draft = recorded_submit
        _parent, run_id = _terminal_subagent_run(runs_db, session.id)
        replacement(rig).on_fleet_drained(  # Wired on the test's loop.
            _drain(session.id, _survivor(run_id, session_id=session.id))
        )
        try:
            assert await _settle(entered.is_set), "the wake never saved its reply row"
            if move == "binding":
                store.rebind_persisted_conversation(session.id, None)
                store.rebind_persisted_conversation(session.id, session.id)
            elif move == "persistence":
                store.persistence = ChatPersistenceService(chacha)
            else:
                ticket = controller.begin_session_close(
                    session.id,
                    expected_revision=controller.lifecycle_impact(
                        session_id=session.id
                    ).revision,
                )
                controller.finalize_session_close(ticket)
                assert all(row.id != session.id for row in store.sessions())
        finally:
            release.set()

        if move == "close":
            assert await _settle(lambda: outcomes, 10), "the wake turn never ended"
            assert adopted_at_entry == [] and gateway.payloads == []
            (outcome,) = outcomes
            assert getattr(outcome, "visible_copy", outcome) == "Session closed.", outcome
            assert outcome.terminal_status is not ConsoleRunStatus.BLOCKED, outcome
            row = chacha.get_connection().execute(
                "SELECT conversation_id, content FROM messages WHERE id=?", (saved[0],)
            ).fetchone()
            assert tuple(row) == (session.id, ""), row
            return

        refused = "The chat changed while the reply was being saved; response was not sent."

        def refusal_rows():
            return [
                row
                for row in store.messages_for_session(session.id)
                if row.role is ConsoleMessageRole.SYSTEM and row.content == refused
            ]

        await _settle(lambda: adopted_at_entry or refusal_rows())
        assert adopted_at_entry in ([], [True]) and False not in adopted_at_entry, (
            f"a wake reached the provider with its reply row unadopted: {adopted_at_entry}"
        )
        assert refusal_rows(), (
            "the moved wake was not refused before the provider: "
            f"{controller.run_state_for(session.id)!r} "
            f"{[r.content for r in store.messages_for_session(session.id) if r.role is ConsoleMessageRole.SYSTEM]}"
        )
        first = store.get_message(saved[0])
        assert first.persisted_message_id is None, "the moved save was adopted"
        row = chacha.get_connection().execute(
            "SELECT conversation_id, content FROM messages WHERE id=?", (saved[0],)
        ).fetchone()
        assert tuple(row) == (session.id, ""), row
    finally:
        await close_rig(rig)
