"""Saved-turn failure policy through actual received runtime custody."""

from __future__ import annotations

import asyncio

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


async def test_received_saved_turn_failure_keeps_original_input_without_dispatch(
    tmp_path, owned_console_databases
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    session = store.sessions()[0]
    session.draft = "original saved turn"
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    db.get_connection().execute(
        "CREATE TRIGGER received_fail_checkpoint "
        "BEFORE INSERT ON console_dispatch_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'received injected failure'); END"
    )
    request = ConsoleTurnCustodyRequest(
        turn_id="received-saved-failure",
        session_id=session.id,
        draft=session.draft,
        configuration=controller.resolve_turn_configuration_snapshot(session.id),
    )
    task = None
    try:
        turn_id = runtime.accept_turn(request)
        assert store.received_turn_for_session(session.id) is not None
        task = runtime._turn_custody[turn_id].task
        with pytest.raises(RuntimeError, match="refused before durable acceptance"):
            await asyncio.wait_for(asyncio.shield(task), 5)
        await asyncio.sleep(0)
        assert gateway.calls == 0
        assert session.draft == request.draft
        assert session.persisted_conversation_id is None
        assert store.received_turn_for_session(session.id) is None
        assert not runtime.has_custodied_turns(session.id)
        (recovery,) = runtime.recoveries_for_session(session.id)
        assert recovery.turn_id == turn_id and recovery.draft == request.draft
        preparation = store.preparation_for_session(session.id)
        assert preparation is not None
        assert preparation.state is ConsoleTurnPreparationState.PAUSED
        assert controller._ordinary_native_commit_tasks(session.id) == ()
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM conversations")
            .fetchone()[0]
            == 0
        )
        assert (
            db.get_connection().execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            == 0
        )
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM console_dispatch_checkpoints")
            .fetchone()[0]
            == 0
        )
    finally:
        if task is not None and not task.done():
            task.cancel()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await runtime.dispose()
