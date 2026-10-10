"""Received and protected fallback inputs obey the physical save outcome."""

import asyncio
from dataclasses import replace

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
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


@pytest.mark.parametrize("route", ["received_intent", "protected_complete_request"])
@pytest.mark.parametrize(
    "outcome", ["normal", "stop", "store_replaced", "newer_identical"]
)
async def test_received_draft_publication_obeys_original_save_outcome(
    tmp_path, monkeypatch, owned_console_databases, route, outcome
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
    inputs = intent.inputs
    if route == "protected_complete_request":
        # A typed carrier only: mounted press/consumer behavior has UI controls.
        request = replace(
            request,
            _pressed_inputs=inputs,
            _pressed_stash=ConsoleDraftStash(
                segments=[], text=inputs.draft, has_paste=False
            ),
            _pressed_attachment_generation=runtime._attached_generation,
        )
    entered, release, exited, calls = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    publications = []
    monkeypatch.setattr(
        runtime,
        "_project_received_input",
        lambda record, *, draft_committed=False: publications.append(record.turn_id),
    )
    successor = None
    newer_revision = None
    task = None
    try:
        if route == "received_intent":
            turn_id = runtime.accept_received_intent(intent)
        else:
            turn_id = runtime.accept_turn(request)
        record = runtime._turn_custody[turn_id]
        assert record.received_claim.draft_revision == inputs.draft_revision
        assert record.store is store
        if route == "protected_complete_request":
            assert record.request is request and record.received_intent is None
        task = record.task
        assert await _until(lambda: entered.is_set() or task.done(), timeout=5)
        if task.done():
            task.result()
        assert entered.is_set()
        assert _saved_state(tmp_path / "controller.sqlite")[0] == [("accepted",)]
        assert gateway.calls == 0 and not exited.is_set()
        assert session.draft == inputs.draft
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
        elif outcome == "newer_identical":
            store.set_session_draft(session.id, "")
            store.set_session_draft(session.id, inputs.draft)
            newer_revision = store.session_input_snapshot(session.id).draft_revision
            assert session.draft == inputs.draft
            assert newer_revision > inputs.draft_revision
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 10)
        assert result.accepted and len(calls) == 1 and exited.is_set()
        assert record.inputs.durable_accepted
        if outcome == "normal":
            assert session.draft == "" and publications == [turn_id]
            assert gateway.calls == 1
        elif outcome == "newer_identical":
            assert session.draft == inputs.draft
            assert (
                store.session_input_snapshot(session.id).draft_revision
                == newer_revision
            )
            # The protected consumer may spend the old capture without clearing
            # new text; an unpressed received intent has no typed projection.
            assert publications == (
                [turn_id] if route == "protected_complete_request" else []
            )
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


@pytest.mark.parametrize("route", ["received_intent", "protected_complete_request"])
async def test_original_sqlite_save_failure_keeps_captured_draft_without_dispatch(
    tmp_path, monkeypatch, owned_console_databases, route
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    session = store.sessions()[0]
    store.set_session_draft(session.id, "keep this failed-save draft")
    runtime = ConsoleRuntime(app=None)
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    configuration = controller.resolve_turn_configuration_snapshot(session.id)

    async def captured_configuration(_session_id, *, selection):
        assert _session_id == session.id
        return configuration

    monkeypatch.setattr(
        controller, "capture_turn_configuration_snapshot", captured_configuration
    )
    request = ConsoleTurnCustodyRequest(
        turn_id="received-sqlite-save-failure",
        session_id=session.id,
        draft=session.draft,
        configuration=configuration,
    )
    intent = _intent((runtime, store, session, request), turn_id=request.turn_id)
    inputs = intent.inputs
    if route == "protected_complete_request":
        request = replace(
            request,
            _pressed_inputs=inputs,
            _pressed_stash=ConsoleDraftStash(
                segments=[], text=inputs.draft, has_paste=False
            ),
            _pressed_attachment_generation=runtime._attached_generation,
        )
    # Permanent, not TEMP: the original commit uses its own worker connection.
    # Aborting its checkpoint insert proves the earlier turn writes roll back.
    db.get_connection().execute(
        "CREATE TRIGGER fail_received_saved_acceptance "
        "BEFORE INSERT ON console_dispatch_checkpoints "
        "BEGIN SELECT RAISE(ABORT, 'received save rollback'); END"
    )
    original_commit = store.commit_durable_turn
    commit_entries, commit_exits = [], []

    def observed_commit(*args, **kwargs):
        commit_entries.append(True)
        try:
            return original_commit(*args, **kwargs)
        finally:
            commit_exits.append(True)

    monkeypatch.setattr(store, "commit_durable_turn", observed_commit)
    publications = []
    monkeypatch.setattr(
        runtime,
        "_project_received_input",
        lambda record, *, draft_committed=False: publications.append(record.turn_id),
    )
    baseline_connections = db.registered_connection_count()
    task = None
    try:
        turn_id = (
            runtime.accept_received_intent(intent)
            if route == "received_intent"
            else runtime.accept_turn(request)
        )
        record = runtime._turn_custody[turn_id]
        task = record.task
        with pytest.raises(
            RuntimeError, match="Console turn was refused before durable acceptance"
        ) as refused:
            await asyncio.wait_for(asyncio.shield(task), 10)
        assert "couldn't save" in refused.value.reason.lower()
        assert commit_entries == commit_exits == [True]
        assert not record.inputs.durable_accepted
        assert gateway.calls == 0 and publications == []
        current = store.session_input_snapshot(session.id)
        assert current.draft == inputs.draft
        assert current.draft_revision == inputs.draft_revision
        assert session.persisted_conversation_id is None
        assert _saved_state(tmp_path / "controller.sqlite") == ([], [])
        assert (
            db.get_connection()
            .execute("SELECT COUNT(*) FROM conversations")
            .fetchone()[0]
            == 0
        )
        assert db.registered_connection_count() == baseline_connections
    finally:
        if task is not None:
            await asyncio.gather(task, return_exceptions=True)
        await runtime.dispose()
