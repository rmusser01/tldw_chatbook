"""Interrupted archive completion and honest retained Library reader feedback."""

import asyncio
from dataclasses import replace
from types import SimpleNamespace

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_conversation_reader_state import (
    ConversationMessageView,
    ConversationReaderState,
    set_conversation_find_query,
)
from tldw_chatbook.UI.Library_Modules.library_conversation_recovery import (
    LibraryConversationRecovery,
)
from tldw_chatbook.UI.Library_Modules.library_conversations_state import (
    LibraryConversationsState,
)
from tldw_chatbook.Widgets.Library.library_conversation_reader import (
    LibraryConversationReader,
)


def loaded_state():
    return ConversationReaderState(
        selected_id="c",
        selected_version=2,
        loaded_id="c",
        loaded_version=2,
        loaded_generation=1,
        generation=1,
        complete=True,
        message_total=1,
        messages=(ConversationMessageView("m", "user", "", "r", 8, "ﬃ needle"),),
    )


def recovery_screen():
    state = LibraryConversationsState(
        reader_state=loaded_state(),
        reader_loaded_metadata={"id": "c", "version": 2, "archived": False},
        reader_selected_metadata={"id": "c", "version": 2, "archived": False},
    )
    paints, reloads = [], []
    screen = SimpleNamespace(
        _conversations_state=state,
        app_instance=SimpleNamespace(),
        query=lambda selector: [True],
        _sync_library_conversation_canvas=lambda: paints.append(True),
        _start_library_conversation_page_request=lambda *args: reloads.append(args),
        _library_conversation_requested_page=1,
        _library_conversation_requested_query="",
    )
    return screen, paints, reloads


@pytest.mark.asyncio
@pytest.mark.parametrize("archived", [True, False])
async def test_completed_change_updates_retained_reader_when_row_leaves_scope(
    monkeypatch, archived
):
    screen, _, reloads = recovery_screen()
    state = screen._conversations_state
    state.reader_loaded_metadata = {"id": "c", "version": 2, "archived": not archived}

    async def write(*args, **kwargs):
        return {"changed": {"c": 3}, "failures": {}}

    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_conversation_recovery.change_conversation_archive",
        write,
    )
    recovery = LibraryConversationRecovery(screen)
    await recovery.change(("c",), archived=archived, expected_versions={"c": 2})
    assert state.reader_loaded_metadata["archived"] is archived
    assert state.reader_loaded_metadata["version"] == 3
    assert state.reader_selected_metadata["version"] == 3
    assert state.reader_state.loaded_version == state.reader_state.selected_version == 3
    assert state.reader_state.loaded_actions_eligible
    assert reloads == [(1, "")]


@pytest.mark.asyncio
@pytest.mark.parametrize("navigate", [False, True])
async def test_cancelled_change_keeps_busy_until_completion_and_retains_receipt(
    monkeypatch, navigate
):
    screen, paints, reloads = recovery_screen()
    started, release = asyncio.Event(), asyncio.Event()

    async def write(*args, **kwargs):
        started.set()
        await release.wait()
        return {"changed": {"c": 3}, "failures": {}}

    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_conversation_recovery.change_conversation_archive",
        write,
    )
    recovery = LibraryConversationRecovery(screen)
    recovery.receipt_versions = {"older": 9}
    operation = asyncio.create_task(
        recovery.change(("c",), archived=True, expected_versions={"c": 2})
    )
    await started.wait()
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    try:
        assert recovery.busy
        if navigate:
            screen.query = lambda selector: []
            state = screen._conversations_state
            state.request_generation += 1
            state.reader_state = replace(
                loaded_state(),
                loaded_id="new",
                selected_id="new",
                loaded_generation=2,
                generation=2,
            )
            state.reader_loaded_metadata = {
                "id": "new",
                "version": 2,
                "archived": False,
            }
        paint_count = len(paints)
    finally:
        release.set()
        await asyncio.wait_for(asyncio.shield(recovery._change_task), 3)
    assert not recovery.busy
    assert recovery.receipt_versions == {"c": 3}
    assert recovery.receipt_copy == "Archived 1 conversation(s)."
    assert reloads == ([] if navigate else [(1, "")])
    if navigate:
        assert len(paints) == paint_count
        assert screen._conversations_state.reader_loaded_metadata["id"] == "new"
        assert screen._conversations_state.reader_loaded_metadata["archived"] is False


@pytest.mark.asyncio
async def test_find_waits_for_mounted_row_and_does_not_claim_normalized_character(
    widget_pilot,
):
    state = set_conversation_find_query(loaded_state(), "needle")
    async with await widget_pilot(
        LibraryConversationReader, state=state, id="reader"
    ) as pilot:
        reader = pilot.app.query_one("#reader", LibraryConversationReader)
        container = reader.query_one("#library-conversation-reader-messages")
        await container.remove_children()
        event = Button.Pressed(
            reader.query_one("#library-conversation-find-next", Button)
        )
        reader.move_find_match(event)
        assert reader._find_navigation_index == -1
        assert not str(
            reader.query_one("#library-conversation-find-position", Static).renderable
        )
        await reader._sync_messages(reader._message_sync_generation)
        assert reader._find_navigation_index == 0
        assert "character" not in str(
            reader.query_one("#library-conversation-find-position", Static).renderable
        )
        assert "Match 1 of 1" in str(
            reader.query_one("#library-conversation-find-position", Static).renderable
        )
        await pilot.pause()
        assert getattr(pilot.app.screen.focused, "message_id", None) == "m"
        assert pilot.app.screen.focused.is_attached


@pytest.mark.asyncio
async def test_unicode_find_uses_stable_message_ordinal(widget_pilot):
    state = set_conversation_find_query(loaded_state(), "needle")
    assert state.find_matches[0].message_offset == 4  # displayed text starts at 2
    async with await widget_pilot(
        LibraryConversationReader, state=state, id="reader"
    ) as pilot:
        reader = pilot.app.query_one("#reader", LibraryConversationReader)
        reader.move_find_match(
            Button.Pressed(reader.query_one("#library-conversation-find-next", Button))
        )
        receipt = str(
            reader.query_one("#library-conversation-find-position", Static).renderable
        )
        assert "message 1" in receipt
        assert "character" not in receipt


@pytest.mark.asyncio
async def test_cancelled_library_worker_observes_real_shared_storage_commit(
    monkeypatch, tmp_path
):
    from tldw_chatbook.Chat import conversation_archive_actions as actions
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

    db = CharactersRAGDB(tmp_path / "cancelled-library.db", client_id="review")
    cid = db.add_conversation({"title": "Survives cancellation"})
    version = db.get_conversation_by_id(cid)["version"]
    local = ChatConversationService(db)
    screen, _, reloads = recovery_screen()
    screen.app_instance.local_chat_conversation_service = local
    started, release = asyncio.Event(), asyncio.Event()
    original = actions.storage_call

    async def held_storage(*args, **kwargs):
        started.set()
        await release.wait()
        return await original(*args, **kwargs)

    monkeypatch.setattr(actions, "storage_call", held_storage)
    recovery = LibraryConversationRecovery(screen)
    operation = asyncio.create_task(
        recovery.change((cid,), archived=True, expected_versions={cid: version})
    )
    try:
        await asyncio.wait_for(started.wait(), 2)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert recovery.busy
        assert cid in screen.app_instance._conversation_archive_inflight
        assert not local.get_conversation_archive_states([cid])[cid]
        release.set()
        await asyncio.wait_for(asyncio.shield(recovery._change_task), 3)
        assert local.get_conversation_archive_states([cid])[cid]
        assert recovery.receipt_versions == {cid: version + 1}
        assert not recovery.busy
        assert reloads == [(1, "")]
        assert cid not in screen.app_instance._conversation_archive_inflight
    finally:
        release.set()
        if recovery._change_task is not None:
            await asyncio.wait_for(asyncio.shield(recovery._change_task), 3)
        db.close_connection()


@pytest.mark.asyncio
async def test_completed_change_does_not_restart_newer_query_or_overwrite_newer_reader(
    monkeypatch,
):
    screen, _, reloads = recovery_screen()
    state = screen._conversations_state

    async def write(*args, **kwargs):
        state.request_generation += 1
        screen._library_conversation_requested_query = "newer query"
        state.reader_state = replace(
            loaded_state(), loaded_version=4, selected_version=4
        )
        state.reader_loaded_metadata = {"id": "c", "version": 4, "archived": False}
        return {"changed": {"c": 3}, "failures": {}}

    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_conversation_recovery.change_conversation_archive",
        write,
    )
    recovery = LibraryConversationRecovery(screen)
    await recovery.change(("c",), archived=True, expected_versions={"c": 2})
    assert recovery.receipt_versions == {"c": 3}
    assert not reloads
    assert state.reader_state.loaded_version == 4
    assert state.reader_loaded_metadata == {"id": "c", "version": 4, "archived": False}


@pytest.mark.asyncio
async def test_pending_find_does_not_focus_a_newer_transcript(widget_pilot):
    state = set_conversation_find_query(loaded_state(), "needle")
    async with await widget_pilot(
        LibraryConversationReader, state=state, id="reader"
    ) as pilot:
        reader = pilot.app.query_one("#reader", LibraryConversationReader)
        await reader.query_one(
            "#library-conversation-reader-messages"
        ).remove_children()
        reader.move_find_match(
            Button.Pressed(reader.query_one("#library-conversation-find-next", Button))
        )
        reader.sync_state(
            replace(
                state,
                loaded_id="new",
                selected_id="new",
                loaded_generation=2,
                generation=2,
            )
        )
        await reader._sync_messages(reader._message_sync_generation)
        await pilot.pause()
        assert reader._find_navigation_index == -1
        assert not str(
            reader.query_one("#library-conversation-find-position", Static).renderable
        )
        assert getattr(pilot.app.screen.focused, "message_id", None) is None


@pytest.mark.asyncio
async def test_completed_change_preserves_new_same_id_reader_request(monkeypatch):
    screen, _, reloads = recovery_screen()
    state = screen._conversations_state

    async def write(*args, **kwargs):
        state.request_generation += 1
        state.reader_state = replace(loaded_state(), generation=2, loading=True)
        return {"changed": {"c": 3}, "failures": {}}

    monkeypatch.setattr(
        "tldw_chatbook.UI.Library_Modules.library_conversation_recovery.change_conversation_archive",
        write,
    )
    recovery = LibraryConversationRecovery(screen)
    await recovery.change(("c",), archived=True, expected_versions={"c": 2})
    assert recovery.receipt_versions == {"c": 3}
    assert state.reader_state.generation == 2
    assert state.reader_state.selected_version == 2
    assert state.reader_state.loading
    assert not reloads
