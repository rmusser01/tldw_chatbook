"""Cancellation must preserve keyboard drafts and committed archive recovery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_console_native_chat_flow import _configure_native_ready_console
from Tests.UI.test_console_send_start_window import _settle_custody
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.Chat import conversation_archive_actions
from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.UI.Console_Modules import archive
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


@pytest.mark.bootstrap_profile
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner_state", ["visible", "background", "closed", "closed-visible"]
)
async def test_cancelled_archive_preflight_restores_only_own_keyboard_draft(
    monkeypatch, owner_state, tmp_path, request
):
    app = _build_test_app()
    database = CharactersRAGDB(tmp_path / "archive-preflight.sqlite", "archive-ui")
    request.addfinalizer(database.close_connection)
    app.chachanotes_db = database
    app.local_chat_conversation_service = ChatConversationService(database)
    _configure_native_ready_console(app)
    host = ConsoleHarness(app)
    entered = asyncio.Event()

    async with host.run_test(size=(120, 36)) as pilot:
        console = host.screen
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        controller = console._ensure_console_chat_controller()
        store = controller.store
        session = store.ensure_session()
        conversation_id = store.persist_session_if_needed(session.id)
        assert conversation_id is not None
        other = store.create_session(title="Other", activate=False)
        store.set_session_draft(other.id, "other tab draft")
        console._session._sync_console_session_draft()
        composer.load_draft("unsent draft")

        async def blocked_gate(_app, checked_conversation_id):
            assert checked_conversation_id == conversation_id
            entered.set()
            await asyncio.Event().wait()

        monkeypatch.setattr(
            conversation_archive_actions, "conversation_send_refusal", blocked_gate
        )
        submit = AsyncMock(side_effect=AssertionError("archive gate was bypassed"))
        monkeypatch.setattr(controller, "submit_draft", submit)
        runtime = console._console_runtime()
        try:
            composer.focus()
            await pilot.press("enter")
            await asyncio.wait_for(entered.wait(), 5)
            records = [
                record
                for record in runtime._turn_custody.values()
                if record.session_id == session.id
            ]
            assert len(records) == 1
            task = records[0].task
            assert task is not None
            assert composer.draft_text() == "unsent draft"
            composer.insert_text(" later")
            assert composer.draft_text() == "unsent draft later"
            if owner_state in {"background", "closed"}:
                console._session.switch_and_bind(other.id)
            if owner_state in {"closed", "closed-visible"}:
                store.close_session(session.id)

            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not runtime.has_custodied_turns(session.id)
            assert conversation_id not in app._conversation_send_inflight
            submit.assert_not_awaited()

            console._session._sync_console_session_draft()
            assert store.session_draft(other.id) == "other tab draft"
            if owner_state == "visible":
                assert composer.draft_text() == "unsent draft later"
            else:
                assert composer.draft_text() == "other tab draft"
            if owner_state in {"closed", "closed-visible"}:
                assert all(item.id != session.id for item in store.sessions())
            else:
                assert store.session_draft(session.id) == "unsent draft later"
        finally:
            await _settle_custody(console)


@pytest.mark.asyncio
@pytest.mark.parametrize("navigate_away", [False, True])
async def test_cancelled_archive_delivers_committed_outcome(monkeypatch, navigate_away):
    started = asyncio.Event()
    complete = asyncio.Event()
    committed = asyncio.Event()
    callbacks = []
    app = SimpleNamespace(notify=Mock())
    screen = SimpleNamespace(
        app_instance=app,
        is_mounted=True,
        _current_console_conversation_id=lambda: "chat-a",
        _workspace=SimpleNamespace(_invalidate_console_persisted_rows_cache=Mock()),
        _sync_native_console_chat_ui=AsyncMock(),
    )

    async def push_screen(modal, *, callback):
        callbacks.append(callback)

    screen.app = SimpleNamespace(screen=screen, push_screen=push_screen)
    monkeypatch.setattr(
        conversation_archive_actions,
        "local_conversation_service",
        lambda _app: object(),
    )
    monkeypatch.setattr(
        conversation_archive_actions,
        "storage_call",
        AsyncMock(return_value={"version": 3, "title": "Chat"}),
    )

    async def change(*args, **kwargs):
        started.set()
        await complete.wait()
        committed.set()
        return {"changed": {"chat-a": 4}, "failures": {}}

    monkeypatch.setattr(
        conversation_archive_actions, "change_conversation_archive", change
    )
    task = asyncio.create_task(archive.archive_current_conversation(screen))
    await asyncio.wait_for(started.wait(), 5)
    if navigate_away:
        screen.app.screen = object()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    complete.set()
    await asyncio.wait_for(committed.wait(), 5)
    pending = tuple(app._console_archive_completion_tasks)
    if pending:
        await asyncio.wait_for(asyncio.gather(*pending), 5)
    screen._workspace._invalidate_console_persisted_rows_cache.assert_called_once()
    if navigate_away:
        assert not callbacks
        assert "Archived chats" in app.notify.call_args.args[0]
        screen._sync_native_console_chat_ui.assert_not_awaited()
    else:
        assert len(callbacks) == 1
        screen._sync_native_console_chat_ui.assert_awaited_once()
        await callbacks[0]("undo")
