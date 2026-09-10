"""Ownership fences across the archive state's asynchronous read boundary."""

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from Tests.UI.test_console_workspace_controller import (
    _browser_row,
    _workspace_controller,
)
from tldw_chatbook.Chat import conversation_archive_actions
from tldw_chatbook.UI.Console_Modules.archive import consume_conversation_resume
from tldw_chatbook.UI.Navigation.pending_handoff_store import (
    ConsoleConversationResumeIntent,
    HandoffChannel,
    PendingHandoffStore,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["refresh", "membership", "archive", "inflight"])
async def test_archive_read_cannot_publish_after_owner_or_archive_changes(
    monkeypatch, change
):
    app = SimpleNamespace(
        local_chat_conversation_service=object(),
        _conversation_archive_states={"c": False},
        _conversation_archive_inflight=set(),
    )
    controller = _workspace_controller(app_instance=app)
    row = _browser_row("c", "Saved chat")

    async def rows(*args, **kwargs):
        return [row], 1, ""

    controller._persisted_console_browser_rows = rows
    entered, release = asyncio.Event(), asyncio.Event()

    async def read(*args, **kwargs):
        entered.set()
        await release.wait()
        return {"c": False}

    monkeypatch.setattr(conversation_archive_actions, "storage_call", read)
    key = ("", None, controller._console_persisted_rows_cache_token)
    controller._console_persisted_rows_refresh_key = key
    task = asyncio.create_task(
        controller._refresh_console_persisted_rows_cache(refresh_key=key)
    )
    await asyncio.wait_for(entered.wait(), 3)
    if change == "refresh":
        controller._console_persisted_rows_refresh_key = ("new query", None, key[2])
    elif change == "membership":
        controller._canonical_membership_revision += 1
    elif change == "archive":
        app._conversation_archive_states["c"] = True
    else:
        app._conversation_archive_inflight.add("c")
    release.set()
    await task
    assert controller._console_persisted_rows_cache is None
    if change == "refresh":
        assert controller._console_persisted_rows_refresh_key[0] == "new query"
    if change == "archive":
        assert app._conversation_archive_states["c"] is True


@pytest.mark.asyncio
async def test_resume_losing_console_visibility_retains_intent_without_activation():
    handoffs = PendingHandoffStore()
    handoffs.stage(
        HandoffChannel.CONSOLE_CONVERSATION_RESUME, ConsoleConversationResumeIntent("c")
    )
    activated = []
    screen = SimpleNamespace(
        app_instance=SimpleNamespace(pending_handoffs=handoffs, notify=Mock()),
        _ensure_console_chat_store=lambda: SimpleNamespace(sessions=list),
    )
    screen.app = SimpleNamespace(screen=screen)

    async def hydrate(cid, *, resume_if, **kwargs):
        screen.app.screen = object()
        if resume_if():
            activated.append(cid)
            return True
        return None

    screen._workspace = SimpleNamespace(_resume_console_workspace_conversation=hydrate)
    await consume_conversation_resume(screen)
    assert activated == []
    assert handoffs.has_pending(HandoffChannel.CONSOLE_CONVERSATION_RESUME)
