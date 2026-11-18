"""Exact result acknowledgement uses existing receipt authority."""

import asyncio
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from types import SimpleNamespace

import pytest

from Tests.Persona_Buddy.test_buddy_inbox import receipt
from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
from tldw_chatbook.Persona_Buddy.interaction import BuddyBinding


def make_app():
    rows = [receipt("one", "live", "saved")]
    acked = []

    def acknowledge(ids):
        acked.extend(ids)
        rows[:] = [r for r in rows if r.activity_id not in ids]
        return len(ids)

    session = ConsoleChatSession(
        id="live", workspace_id="ws", persisted_conversation_id="saved"
    )
    receipts = SimpleNamespace(
        unseen_snapshot=lambda: tuple(rows), acknowledge=acknowledge
    )
    from tldw_chatbook.Chat.console_chat_models import ConsoleRunState

    controller = SimpleNamespace(
        run_state_for=lambda _: ConsoleRunState(),
        activity_for=lambda _: SimpleNamespace(),
    )
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            chat_store=SimpleNamespace(sessions=lambda: (session,)),
            chat_controller=controller,
            activity_receipts=receipts,
        ),
        workspace_registry_service=SimpleNamespace(
            get_workspace=lambda _: SimpleNamespace(
                name="Workspace", archived=False, authority="local-only"
            ),
            list_workspace_conversations=lambda _: (),
        ),
    )
    return app, rows, acked, session


@pytest.mark.asyncio
async def test_acknowledge_one_frozen_result_leaves_newer_result_unseen():
    from tldw_chatbook.UI.Navigation.buddy_workspace import BuddyWorkspaceCoordinator

    app, rows, acked, _ = make_app()
    coordinator = BuddyWorkspaceCoordinator(
        app, BuddyBinding(kind="workspace", target_id="ws")
    )
    _, entries = await coordinator.snapshot()
    rows.append(receipt("two", "live", "saved"))
    assert await coordinator.acknowledge(entries[0]) == 1
    assert acked == ["one"]
    assert [r.activity_id for r in rows] == ["two"]


@pytest.mark.asyncio
async def test_moved_conversation_cannot_acknowledge_old_workspace_row():
    from tldw_chatbook.UI.Navigation.buddy_workspace import BuddyWorkspaceCoordinator

    app, _, acked, session = make_app()
    coordinator = BuddyWorkspaceCoordinator(
        app, BuddyBinding(kind="workspace", target_id="ws")
    )
    _, entries = await coordinator.snapshot()
    session.workspace_id = "elsewhere"
    with pytest.raises(ValueError, match="no longer"):
        await coordinator.acknowledge(entries[0])
    assert acked == []


@pytest.mark.asyncio
async def test_storage_error_opening_result_does_not_escape_or_acknowledge():
    from tldw_chatbook.UI.Navigation.buddy_workspace import BuddyWorkspaceCoordinator

    app, _, acked, _ = make_app()
    notices = []
    app.notify = lambda message, **kwargs: notices.append((message, kwargs))
    coordinator = BuddyWorkspaceCoordinator(
        app, BuddyBinding(kind="workspace", target_id="ws")
    )
    _, entries = await coordinator.snapshot()

    def unavailable(_):
        raise OSError("private storage path")

    app.workspace_registry_service.get_workspace = unavailable
    await coordinator._open_entry(entries[0])
    assert acked == []
    assert len(notices) == 1
    assert notices[0][1]["severity"] == "error"
    assert "private storage path" not in notices[0][0]


@pytest.mark.asyncio
@pytest.mark.bootstrap_profile
@pytest.mark.parametrize("borrowed", [False, True], ids=["cold", "warm-borrower"])
@pytest.mark.parametrize("failure", [False, True], ids=["result", "read-error"])
async def test_inbox_membership_read_retires_only_its_native_handles(
    tmp_path, monkeypatch, borrowed, failure
):
    from tldw_chatbook.Chat.chat_conversation_service import ChatConversationService
    from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.UI.Navigation.buddy_workspace import BuddyWorkspaceCoordinator
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    chat = CharactersRAGDB(tmp_path / "chat.db", "inbox-native")
    workspace = WorkspaceDB(tmp_path / "workspace.db")
    registry = LocalWorkspaceRegistryService(workspace)
    service = ChatConversationService(chat)
    saved = chat.add_conversation({"title": "Saved reply"})
    registry.create_workspace(workspace_id="ws", name="Workspace")
    registry.link_membership("ws", item_type="conversation", item_id=saved)
    chat.close_connection()
    workspace.close()
    app, _, _, _ = make_app()
    app.workspace_registry_service = registry
    app.local_chat_conversation_service = service
    app.console_runtime.activity_receipts.unseen_snapshot = lambda: (
        receipt("one", None, saved),
    )
    coordinator = BuddyWorkspaceCoordinator(
        app, BuddyBinding(kind="workspace", target_id="ws")
    )
    opened = []

    def observe(factory):
        def acquire():
            connection = factory()
            if all(connection is not previous for previous in opened):
                opened.append(connection)
            return connection

        return acquire

    monkeypatch.setattr(chat, "get_connection", observe(chat.get_connection))
    monkeypatch.setattr(
        workspace, "_get_connection", observe(workspace._get_connection)
    )
    original_read = service.get_conversation_metadata

    def read(conversation_id):
        result = original_read(conversation_id)
        if failure:
            raise OSError("controlled inbox read failure")
        return result

    monkeypatch.setattr(service, "get_conversation_metadata", read)
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:

        async def on_worker(operation, *args, **kwargs):
            return await loop.run_in_executor(
                executor, partial(operation, *args, **kwargs)
            )

        with monkeypatch.context() as patch:
            patch.setattr(asyncio, "to_thread", on_worker)

            def warm():
                chat.get_connection()
                workspace._held_connection()

            def assert_native():
                assert len(opened) == 2
                for connection in opened:
                    if borrowed:
                        assert connection.execute("SELECT 1").fetchone()[0] == 1
                    else:
                        with pytest.raises(
                            sqlite3.ProgrammingError, match="closed database"
                        ):
                            connection.execute("SELECT 1")

            def cleanup():
                for connection in opened:
                    try:
                        connection.rollback()
                    except sqlite3.ProgrammingError:
                        pass
                chat.close_connection()
                workspace.close()

            try:
                if borrowed:
                    await on_worker(warm)
                if failure:
                    with pytest.raises(OSError, match="controlled inbox read failure"):
                        await coordinator.snapshot()
                else:
                    title, entries = await coordinator.snapshot()
                    assert title == "Workspace"
                    assert any(entry.title == "Saved reply" for entry in entries)
                await on_worker(assert_native)
            finally:
                await on_worker(cleanup)
