"""Task-19576: STTS AudioBook "Import from Notes"/"Import from Conversation"
used to crash with an uncaught `ImportError`.

`_import_from_notes` imported `fetch_all_notes`/`fetch_note_by_id` and
`_import_from_conversation` imported `fetch_all_conversations`/
`fetch_messages_by_conversation_id`, all four from
`tldw_chatbook.DB.ChaChaNotes_DB` -- none of which exist there (a regex for
``def fetch_all_notes`` etc. returns zero matches). The imports sat OUTSIDE
the ``try:`` blocks in both methods, so the resulting `ImportError`
propagated on every use, on the live, routed path: `stts` route ->
`STTSScreen` -> rail entry "AudioBook/Podcast" -> the "Import From"
Select's Notes/Conversation options -> `#import-content-btn` ->
`_import_content` -> `_import_from_notes`/`_import_from_conversation`.

Born-red: every test below that calls `_import_from_notes()`/
`_import_from_conversation()` directly (no `pytest.raises` wrapper) fails
at base with that exact `ImportError` propagating out of the call --
`cannot import name 'fetch_all_notes' from
'tldw_chatbook.DB.ChaChaNotes_DB'`, and similarly for the other three
names. They pass once both methods route through the live
`notes_scope_service`/`chat_conversation_scope_service` seams instead.
"""

from __future__ import annotations

from typing import Any

import pytest
from textual.app import App, ComposeResult
from textual.widgets import TextArea

from tldw_chatbook.UI.STTS_Window import AudioBookGenerationWidget


class _Host(App[None]):
    def compose(self) -> ComposeResult:
        yield AudioBookGenerationWidget()


class _FakeNotesScopeService:
    """Minimal stand-in for `notes_scope_service`, matching the async
    `list_notes(*, scope, user_id, limit=..., offset=...)` contract."""

    def __init__(self, notes: list[dict[str, Any]]) -> None:
        self._notes = notes

    async def list_notes(self, *, scope, user_id, limit: int = 100, offset: int = 0):
        return list(self._notes)


class _FakeConversationScopeService:
    """Minimal stand-in for `chat_conversation_scope_service`, matching the
    async `list_conversations`/`get_messages_with_context` contracts."""

    def __init__(
        self,
        items: list[dict[str, Any]],
        messages_by_id: dict[str, list[dict[str, Any]]],
    ) -> None:
        self._items = items
        self._messages_by_id = messages_by_id

    async def list_conversations(self, *, mode: str = "local", **kwargs: Any):
        limit = kwargs.get("limit", 100)
        return {
            "items": list(self._items),
            "pagination": {
                "limit": limit,
                "offset": 0,
                "total": len(self._items),
                "has_more": False,
            },
        }

    async def get_messages_with_context(
        self,
        conversation_id: str,
        *,
        mode: str = "server",
        limit: int = 200,
        offset: int = 0,
        include_rag_context: bool = True,
    ):
        messages = self._messages_by_id.get(conversation_id, [])
        return messages[offset : offset + limit]


@pytest.mark.asyncio
async def test_import_from_notes_no_longer_crashes_and_loads_real_note_content():
    """Primary born-red for the notes path (see module docstring)."""
    app = _Host()
    app.notes_scope_service = _FakeNotesScopeService(
        [
            {
                "id": "note-1",
                "title": "My Note",
                "content": "Hello from a real note.",
                "created_at": "2026-01-01",
            }
        ]
    )
    app.notes_user_id = "default_user"

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)

        # This line is where the base code raises ImportError.
        widget._import_from_notes()
        await pilot.pause()

        # NOTE: the worker is suspended awaiting the dialog's dismissal
        # (`wait_for_dismiss=True`), so `workers.wait_for_complete()` must
        # not be called until AFTER `dismiss()` below -- calling it first
        # would deadlock forever waiting on a worker that is waiting on us.
        assert type(app.screen).__name__ == "NoteSelectionDialog"
        app.screen.dismiss(["note-1"])
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert "Hello from a real note." in widget.content_text
        content_preview = app.query_one("#content-preview", TextArea)
        assert content_preview.disabled is False


@pytest.mark.asyncio
async def test_import_from_notes_notifies_instead_of_crashing_when_unavailable():
    """Worst case: no `notes_scope_service` wired at all (a bare host, as
    used elsewhere in this test module). At base this still raises
    ImportError; after the fix it degrades to a notify."""
    app = _Host()
    notifications: list[tuple[Any, dict[str, Any]]] = []

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        app.notify = lambda *a, **k: notifications.append((a, k))
        widget = app.query_one(AudioBookGenerationWidget)

        widget._import_from_notes()
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert notifications, "expected a graceful notify, not a crash"
        assert widget.is_mounted


@pytest.mark.asyncio
async def test_import_from_conversation_no_longer_crashes_and_loads_real_messages():
    """Primary born-red for the conversation path (see module docstring)."""
    app = _Host()
    app.chat_conversation_scope_service = _FakeConversationScopeService(
        items=[
            {
                "id": "conv-1",
                "title": "Chat",
                "message_count": 2,
                "created_at": "2026-01-01",
                "last_modified": "2026-01-02",
            }
        ],
        messages_by_id={
            "conv-1": [
                {"role": "user", "content": "Hi there"},
                {
                    "role": "assistant",
                    "content": "Hello! Real conversation content.",
                },
            ]
        },
    )

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)

        # This line is where the base code raises ImportError.
        widget._import_from_conversation()
        await pilot.pause()

        # See the notes test above: do not wait_for_complete() before dismiss.
        assert type(app.screen).__name__ == "ConversationSelectionDialog"
        app.screen.dismiss(
            {
                "conversation_id": "conv-1",
                "include_all": True,
                "include_user": False,
                "include_assistant": False,
                "include_speakers": True,
            }
        )
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert "Hi there" in widget.content_text
        assert "Real conversation content." in widget.content_text


@pytest.mark.asyncio
async def test_import_from_conversation_role_filter_is_case_insensitive():
    """Regression: the removed code compared `role != "user"` against
    ``sender`` values the DB actually stores capitalized ("User"), so
    "User messages only"/"Assistant messages only" could never match
    anything. The rewritten filter normalizes case."""
    app = _Host()
    app.chat_conversation_scope_service = _FakeConversationScopeService(
        items=[
            {
                "id": "conv-1",
                "title": "Chat",
                "message_count": 2,
                "created_at": "2026-01-01",
                "last_modified": "2026-01-02",
            }
        ],
        messages_by_id={
            "conv-1": [
                {"role": "User", "content": "user turn"},
                {"role": "Assistant", "content": "assistant turn"},
            ]
        },
    )

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        widget = app.query_one(AudioBookGenerationWidget)

        widget._import_from_conversation()
        await pilot.pause()

        app.screen.dismiss(
            {
                "conversation_id": "conv-1",
                "include_all": False,
                "include_user": True,
                "include_assistant": False,
                "include_speakers": False,
            }
        )
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert "user turn" in widget.content_text
        assert "assistant turn" not in widget.content_text


@pytest.mark.asyncio
async def test_import_from_conversation_notifies_instead_of_crashing_when_unavailable():
    app = _Host()
    notifications: list[tuple[Any, dict[str, Any]]] = []

    async with app.run_test(size=(120, 48)) as pilot:
        await pilot.pause()
        app.notify = lambda *a, **k: notifications.append((a, k))
        widget = app.query_one(AudioBookGenerationWidget)

        widget._import_from_conversation()
        await pilot.pause()
        await app.workers.wait_for_complete()

        assert notifications, "expected a graceful notify, not a crash"
        assert widget.is_mounted
