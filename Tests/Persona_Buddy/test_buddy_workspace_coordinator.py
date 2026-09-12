"""Exact result acknowledgement uses existing receipt authority."""

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
