"""Production-shaped Active/History projection contracts for Ctrl+K."""

from __future__ import annotations

import asyncio
from threading import Event
from types import SimpleNamespace

import pytest

from Tests.UI.test_console_workspace_controller import _workspace_controller
from tldw_chatbook.Chat.console_switcher_state import (
    ActivityGroup,
    ConsoleSwitcherEntry,
)
from tldw_chatbook.Workspaces.conversation_browser_state import (
    ConsoleConversationBrowserInputRow,
)


def _native_row(session_id: str = "session-1") -> ConsoleConversationBrowserInputRow:
    return ConsoleConversationBrowserInputRow(
        row_key=f"native:{session_id}",
        conversation_id=None,
        native_session_id=session_id,
        title="Live agent work",
        scope_type="workspace",
        workspace_id="workspace-1",
        workspace_label="Workspace 1",
        status="active session",
        selected=True,
        source_kind="native",
        updated_sort="2026-08-23T12:00:00+00:00",
        run_marker="[*]",
    )


class _ReceiptSnapshot:
    def __init__(self, *, state: str = "ready") -> None:
        self._state = state

    def unseen_snapshot(self):
        return ()

    def hydration_state(self):
        return self._state


def _projection_controller(app):
    controller = _workspace_controller(app_instance=app)
    controller._native_console_browser_rows = lambda _current=None: [_native_row()]
    controller._membership_console_browser_rows = lambda _current=None: []
    return controller


@pytest.mark.asyncio
async def test_active_is_immediate_while_bounded_history_is_blocked():
    entered = Event()
    release = Event()

    def list_conversations(**_kwargs):
        entered.set()
        assert release.wait(5)
        return {"items": [], "pagination": {"total": 0}}

    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=_ReceiptSnapshot(),
        ),
        local_chat_conversation_service=SimpleNamespace(
            list_conversations=list_conversations
        ),
    )
    controller = _projection_controller(app)

    history = asyncio.create_task(
        controller.load_console_session_switcher_history(
            query="", offset=0, limit=50
        )
    )
    assert await asyncio.to_thread(entered.wait, 5)

    active = controller.console_session_switcher_active_entries()

    assert len(active) == 1
    assert isinstance(active[0], ConsoleSwitcherEntry)
    assert active[0].group is ActivityGroup.WORKING
    release.set()
    assert (await history).entries == ()


@pytest.mark.asyncio
async def test_history_uses_one_all_local_bounded_page_with_explicit_targets():
    calls: list[dict[str, object]] = []

    def list_conversations(**kwargs):
        calls.append(kwargs)
        items = [
            {
                "id": f"conversation-{index}",
                "title": f"Conversation {index}",
                "scope_type": "workspace",
                "workspace_id": "workspace-1",
                "state": "in-progress",
                "last_modified": "2026-08-23T12:00:00+00:00",
            }
            for index in range(70)
        ]
        return {"items": items, "pagination": {"total": 70}}

    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=_ReceiptSnapshot(),
        ),
        local_chat_conversation_service=SimpleNamespace(
            list_conversations=list_conversations
        ),
    )
    controller = _projection_controller(app)

    page = await controller.load_console_session_switcher_history(
        query="release", offset=0, limit=500
    )

    assert len(page.entries) == 50
    assert page.total == 70
    assert page.has_more is True
    assert calls == [
        {
            "query": "release",
            "scope_type": "all",
            "limit": 50,
            "offset": 0,
        }
    ]
    assert all(entry.target is not None for entry in page.entries)
    assert all(entry.row_key.startswith("conversation:profile-a:") for entry in page.entries)


@pytest.mark.asyncio
async def test_receipt_degradation_leaves_open_active_and_history_available():
    def list_conversations(**_kwargs):
        return {
            "items": [
                {
                    "id": "saved-1",
                    "title": "Saved conversation",
                    "scope_type": "global",
                    "last_modified": "2026-08-22T12:00:00+00:00",
                }
            ],
            "pagination": {"total": 1},
        }

    receipts = _ReceiptSnapshot(state="degraded")
    app = SimpleNamespace(
        console_runtime=SimpleNamespace(
            profile_authority="profile-a",
            authority_token="runtime-a",
            activity_receipts=receipts,
        ),
        local_chat_conversation_service=SimpleNamespace(
            list_conversations=list_conversations
        ),
    )
    controller = _projection_controller(app)
    tree_before = controller._workspace_tree_search
    pages_before = dict(controller._workspace_page_attempts)

    active = controller.console_session_switcher_active_entries()
    history = await controller.load_console_session_switcher_history(
        query="saved", offset=0, limit=50
    )

    assert receipts.hydration_state() == "degraded"
    assert active and active[0].title == "Live agent work"
    assert [entry.title for entry in history.entries] == ["Saved conversation"]
    assert controller._workspace_tree_search is tree_before
    assert controller._workspace_page_attempts == pages_before
