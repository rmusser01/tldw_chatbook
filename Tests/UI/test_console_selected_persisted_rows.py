"""Persisted browser reads retire before obsolete visits discard publication."""

import asyncio
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_selected_refresh_coverage import _cover, _visit
from Tests.UI.test_console_workspace_controller import (
    _browser_row,
    _workspace_controller,
)
from tldw_chatbook.Chat import conversation_archive_actions
from tldw_chatbook.Chat.console_execution_process import ConsoleChannelLost
from tldw_chatbook.UI.Console_Modules import execution_decisions, execution_state


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["rebind", "refresh_key"])
async def test_queued_original_partial_keeps_scheduling_visit_and_refresh_key(
    monkeypatch, change
):
    screen, client, app = _visit(monkeypatch)
    queued = []
    screen.run_worker = lambda work, **_: queued.append(work)
    controller = _workspace_controller(screen=screen, app_instance=app)

    async def unexpected(*_, **__):
        raise AssertionError("An obsolete queued refresh issued a read")

    controller._persisted_console_browser_rows = unexpected
    controller._sync_persisted_console_browser_rows("visit A query")
    assert len(queued) == 1
    old_key = controller._console_persisted_rows_refresh_key
    if change == "rebind":
        client._visit = screen._console_process_visit = 2
    else:
        controller._console_persisted_rows_refresh_key = (
            "visit B query",
            None,
            old_key[2],
        )
    assert await queued[0]() is None
    assert controller._console_persisted_rows_cache is None
    assert controller._console_persisted_rows_refresh_key == (
        None if change == "rebind" else ("visit B query", None, old_key[2])
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["before", "rows", "archive"])
@pytest.mark.parametrize("transition", ["covered", "remounted"])
async def test_obsolete_persisted_refresh_never_reads_or_publishes_for_new_visit(
    monkeypatch,
    boundary,
    transition,
):
    screen, client, app = _visit(monkeypatch)
    app._conversation_archive_states = {"saved": True}
    calls = []
    row = _browser_row("saved", "Saved chat")

    def change_visit():
        if transition == "covered":
            _cover(screen, app)
        else:
            client._visit += 1
            # Before entry the old screen has not been rebound yet.
            if boundary != "before":
                screen._console_process_visit = client._visit

    def store():
        assert screen.is_current
        calls.append("store")
        return SimpleNamespace(
            sessions=lambda: [SimpleNamespace(persisted_conversation_id="saved")]
        )

    controller = _workspace_controller(
        screen=screen,
        app_instance=app,
        current_chat_store_accessor=store,
        sync_workspace_context=lambda: calls.append("paint"),
    )
    previous = ([_browser_row("previous", "Previous")], 1, "")
    controller._console_persisted_rows_cache = previous
    key = ("new query", None, controller._console_persisted_rows_cache_token)
    controller._console_persisted_rows_refresh_key = key

    async def rows(*_, **__):
        calls.append("rows")
        try:
            await asyncio.sleep(0)
            if boundary == "rows":
                change_visit()
            return [row], 1, ""
        finally:
            calls.append("rows returned")

    async def archive(*_, **__):
        calls.append("archive")
        try:
            await asyncio.sleep(0)
            change_visit()
            return {"saved": False}
        finally:
            calls.append("archive returned")

    controller._persisted_console_browser_rows = rows
    monkeypatch.setattr(
        conversation_archive_actions, "local_conversation_service", lambda _: object()
    )
    monkeypatch.setattr(conversation_archive_actions, "storage_call", archive)
    if boundary == "before":
        change_visit()
    result = await controller._refresh_console_persisted_rows_cache(
        "new query", refresh_key=key
    )
    assert result is None if boundary == "before" else result == ([row], 1, "")
    assert (
        calls
        == {
            "before": [],
            "rows": ["rows", "rows returned"],
            "archive": [
                "rows",
                "rows returned",
                "store",
                "archive",
                "archive returned",
            ],
        }[boundary]
    )
    assert controller._console_persisted_rows_cache is previous
    assert app._conversation_archive_states == {"saved": True}
    assert controller._console_persisted_rows_refresh_key is None


@pytest.mark.asyncio
@pytest.mark.parametrize("owned_modal", [False, True])
@pytest.mark.parametrize("failure", [False, True])
async def test_current_persisted_refresh_retains_publication_and_active_failures(
    monkeypatch,
    owned_modal,
    failure,
):
    screen, client, app = _visit(monkeypatch)
    if owned_modal:
        _cover(screen, app)
        monkeypatch.setattr(
            execution_decisions, "is_owned_project_modal", lambda *_: True
        )
    controller = _workspace_controller(screen=screen, app_instance=app)
    result = ([_browser_row("saved", "Saved")], 1, "")
    error = ConsoleChannelLost("active channel failure")

    def checked_client(_):
        if client._channel_error is not None:
            raise client._channel_error
        return client

    monkeypatch.setattr(execution_state, "selected_client", checked_client)

    async def rows(*_, **__):
        await asyncio.sleep(0)
        if failure:
            client._channel_error = error
        return result

    controller._persisted_console_browser_rows = rows
    if failure:
        with pytest.raises(ConsoleChannelLost) as caught:
            await controller._refresh_console_persisted_rows_cache()
        assert caught.value is error
        assert controller._console_persisted_rows_cache is None
    else:
        assert await controller._refresh_console_persisted_rows_cache() == result
        assert controller._console_persisted_rows_cache == result
