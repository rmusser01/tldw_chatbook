"""Received-input publication obeys the original physical save outcome."""

import asyncio

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller, _until
from Tests.Chat.test_console_native_commit_integration import (
    _hold_original_commit,
    _saved_state,
)
from Tests.Chat.test_console_received_intent_custody import _intent
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


@pytest.mark.parametrize("outcome", ["normal", "stop", "store_replaced"])
async def test_received_draft_publication_obeys_original_save_outcome(
    tmp_path, monkeypatch, owned_console_databases, outcome
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    session = store.sessions()[0]
    store.set_session_draft(session.id, "received saved draft")
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    configuration = controller.resolve_turn_configuration_snapshot(session.id)

    # Configuration is the original controller's complete result. This control
    # isolates saved acceptance; configuration source ownership has separate tests.
    async def captured_configuration(_session_id, *, selection):
        assert _session_id == session.id
        return configuration

    monkeypatch.setattr(
        controller, "capture_turn_configuration_snapshot", captured_configuration
    )
    request = ConsoleTurnCustodyRequest(
        turn_id="received-save-outcome",
        session_id=session.id,
        draft=session.draft,
        configuration=configuration,
    )
    intent = _intent((runtime, store, session, request), turn_id=request.turn_id)
    entered, release, exited, calls = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    publications = []
    monkeypatch.setattr(
        runtime,
        "_project_received_input",
        lambda record: publications.append(record.turn_id),
    )
    successor = None
    task = None
    try:
        runtime.accept_received_intent(intent)
        record = runtime._turn_custody[intent.turn_id]
        task = record.task
        assert await _until(lambda: entered.is_set() or task.done(), timeout=5)
        if task.done():
            task.result()
        assert entered.is_set()
        assert _saved_state(tmp_path / "controller.sqlite")[0] == [("accepted",)]
        assert gateway.calls == 0 and not exited.is_set()
        if outcome == "stop":
            assert controller.stop_active_run(record_user_stop=False)
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0)
                assert not task.done()
        elif outcome == "store_replaced":
            successor = ConsoleChatStore()
            next_session = successor.create_session(
                session_id=session.id, title="Successor"
            )
            successor.set_session_draft(next_session.id, intent.inputs.draft)
            controller.store = successor
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 10)
        assert result.accepted and len(calls) == 1 and exited.is_set()
        assert record.inputs.durable_accepted
        if outcome == "normal":
            assert session.draft == "" and publications == [intent.turn_id]
            assert gateway.calls == 1
        else:
            assert session.draft == intent.inputs.draft
            assert publications == [] and gateway.calls == 0
            assert not result.should_clear_draft
            if successor is not None:
                assert next_session.draft == intent.inputs.draft
                assert next_session.persisted_conversation_id is None
                assert successor.messages_for_session(next_session.id) == []
    finally:
        release.set()
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        controller.store = store
        await runtime.dispose()
