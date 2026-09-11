"""Cancellation must preserve keyboard drafts and committed archive recovery."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_destination_shells import _wait_for_selector
from Tests.UI.test_product_maturity_gate1_core_loop_screen_adaptation import (
    ConsoleHarness,
)
from tldw_chatbook.UI.Console_Modules import archive
from tldw_chatbook.Widgets.Console import ConsoleComposerBar


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "owner_state", ["visible", "background", "closed", "closed-visible"]
)
async def test_cancelled_archive_preflight_restores_only_own_keyboard_draft(
    monkeypatch, owner_state
):
    app = _build_test_app()
    host = ConsoleHarness(app)
    entered = asyncio.Event()

    async def blocked_gate(*_args):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(
        "tldw_chatbook.Chat.conversation_archive_actions.conversation_send_refusal",
        blocked_gate,
    )
    async with host.run_test(size=(120, 36)) as pilot:
        console = host.screen
        await _wait_for_selector(console, pilot, "#console-native-composer")
        composer = console.query_one("#console-native-composer", ConsoleComposerBar)
        controller = console._ensure_console_chat_controller()
        session = controller.store.ensure_session()
        controller.store.switch_session(session.id)
        console._session._sync_console_session_draft()
        composer.load_draft("unsent draft")
        stash = composer.stash_draft_for_send()
        console._console_inflight_send_stashes[session.id] = stash
        task = asyncio.create_task(
            console._submit_console_native_draft_observed(stash.text, session.id)
        )
        await asyncio.wait_for(entered.wait(), 5)
        if owner_state in {"background", "closed"}:
            other = controller.store.create_session(title="Other")
            controller.store.switch_session(other.id)
            console._console_visible_draft_session_id = other.id
            controller.store.set_session_draft(session.id, " later")
            composer.load_draft("other tab draft")
            if owner_state == "closed":
                controller.store.close_session(session.id)
        else:
            composer.insert_text(" later")
            if owner_state == "closed-visible":
                controller.store.close_session(session.id)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert session.id not in console._console_inflight_send_stashes
        assert not console._console_submit_session_by_task
        if owner_state in {"background", "closed"}:
            assert composer.draft_text() == "other tab draft"
            if owner_state == "closed":
                assert all(
                    item.id != session.id for item in controller.store.sessions()
                )
            else:
                assert (
                    controller.store.session_draft(session.id) == "unsent draft later"
                )
        elif owner_state == "closed-visible":
            assert composer.draft_text() == " later"
            assert all(item.id != session.id for item in controller.store.sessions())
        else:
            assert composer.draft_text() == "unsent draft later"


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
    monkeypatch.setattr(archive, "local_conversation_service", lambda _app: object())
    monkeypatch.setattr(
        archive, "storage_call", AsyncMock(return_value={"version": 3, "title": "Chat"})
    )

    async def change(*args, **kwargs):
        started.set()
        await complete.wait()
        committed.set()
        return {"changed": {"chat-a": 4}, "failures": {}}

    monkeypatch.setattr(archive, "change_conversation_archive", change)
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
