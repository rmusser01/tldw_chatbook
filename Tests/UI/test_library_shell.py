"""Library shell (L1) rail + conversations canvas pilot contracts."""

import asyncio
import json
import re
import threading
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.widgets import Button, Collapsible, Input, Markdown, Static, TextArea

from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_NOTE_ID,
    LIBRARY_NAV_CONTEXT_NOTES_CREATE,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_CREATE_NOTE,
)
from tldw_chatbook.Third_Party.textual_fspicker import FileOpen, FileSave
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
    StaticLibraryNotesScopeService,
    _link_library_items_to_active_workspace,
)
from Tests.UI.test_library_content_hub import StaticLibraryCollectionsService
from Tests.UI.test_screen_navigation import _build_test_app

LIBRARY_TEST_SIZE = (170, 48)


class LibraryHarness(App):
    """Mount a single LibraryScreen with the real app stylesheet."""

    CSS_PATH = str(
        Path(__file__).resolve().parents[2]
        / "tldw_chatbook"
        / "css"
        / "tldw_cli_modular.tcss"
    )

    def __init__(self, app_instance, seen_routes=None, screen=None):
        super().__init__()
        self.app_instance = app_instance
        self.seen_routes = seen_routes if seen_routes is not None else []
        self.seen_contexts = []
        self._screen = screen

    async def on_mount(self) -> None:
        await self.push_screen(self._screen or LibraryScreen(self.app_instance))

    def on_navigate_to_screen(self, message) -> None:
        self.seen_routes.append(message.screen_name)
        self.seen_contexts.append(dict(message.screen_context or {}))


def _active_library_screen(host: LibraryHarness):
    return host.screen_stack[-1]


def _visible_text(screen) -> str:
    chunks = []
    for widget in screen.query("Static"):
        renderable = getattr(widget, "renderable", "")
        chunks.append(getattr(renderable, "plain", str(renderable)))
    for widget in screen.query("Button"):
        label = getattr(widget, "label", "")
        chunks.append(str(label) if label is not None else "")
    return " ".join(chunks)


def _seed_conversations(app, conversations, *, notes=None, media=None, highlights=None):
    app.notes_scope_service = StaticLibraryNotesScopeService(notes or [])
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        media or [], highlights=highlights
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        conversations
    )


async def _wait_for_library_shell(screen, pilot, *, attempts=120):
    for _ in range(attempts):
        if getattr(screen, "_library_loaded", False) and screen.query("#library-rail"):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.02)
    raise AssertionError(
        f"Library shell never loaded. Visible text: {_visible_text(screen)}"
    )


async def _wait_for_selector(screen, pilot, selector, *, attempts=120):
    for _ in range(attempts):
        matches = list(screen.query(selector))
        if matches:
            await pilot.pause()
            return matches[0]
        await pilot.pause(0.02)
    raise AssertionError(
        f"{selector} never mounted. Visible text: {_visible_text(screen)}"
    )


def _two_conversations():
    return [
        {
            "title": "Quarterly planning sync",
            "conversation_id": "chat-1",
            "message_count": 8,
            "updated_at": "2026-06-01T10:00:00Z",
        },
        {
            "title": "Design review notes",
            "conversation_id": "chat-2",
            "message_count": 3,
            "updated_at": "2026-06-02T09:30:00Z",
        },
    ]


def _two_media_items():
    return [
        {
            "id": "media-1",
            "title": "Interview Recording",
            "type": "audio",
            "last_modified": "2026-07-06T08:00:00Z",
            "author": "Jordan Lee",
            "keywords": ["interview", "audio"],
            "content": "Full transcript: the interview recording covers the quarterly roadmap.",
            "version": 1,
        },
        {
            "id": "media-2",
            "title": "Product Demo Video",
            "type": "video",
            "last_modified": "2026-07-06T10:00:00Z",
            "author": "Morgan Lee",
            "keywords": ["demo", "video"],
            "content": "Full transcript: the product demo video walks through the new dashboard.",
            "version": 2,
        },
    ]


@pytest.mark.asyncio
async def test_library_shell_renders_rail_sections_and_landing_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        header = str(screen.query_one("#library-header-line").renderable)
        assert header == "Library | Local"

        for selector in (
            "#library-rail-section-header-browse",
            "#library-rail-section-header-create",
            "#library-rail-section-header-ingest",
            "#library-rail-section-header-details",
        ):
            assert screen.query_one(selector)

        visible = _visible_text(screen)
        assert "Conversations (2)" in visible
        assert "Search, pick a content type, or ingest something new." in visible
        assert screen.query_one("#library-canvas-landing")

        assert not screen.query("#library-mode-bar")
        assert not screen.query("#library-contract-grid")
        assert not screen.query("#library-notes-summary")


@pytest.mark.asyncio
async def test_library_shell_browse_conversations_renders_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-0")

        rows = list(screen.query(".library-conversation-row"))
        assert len(rows) == 2
        first_label = str(rows[0].label)
        assert first_label.startswith("▸")

        preview = str(screen.query_one("#library-conversation-preview-lines").renderable)
        assert "Messages:" in preview
        assert screen.query_one("#library-conversation-open-console")


@pytest.mark.asyncio
async def test_library_shell_conversation_row_switches_selection():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-row-1")

        # Rows sort newest-first: chat-2 (06-02) is row 0, chat-1 (06-01) is row 1.
        preview_before = str(
            screen.query_one("#library-conversation-preview-lines").renderable
        )
        assert "Design review notes" in preview_before
        assert screen._selected_conversation_id == "chat-2"

        screen.query_one("#library-conversation-row-1").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._selected_conversation_id == "chat-1"
        preview_after = str(
            screen.query_one("#library-conversation-preview-lines").renderable
        )
        assert "Quarterly planning sync" in preview_after


@pytest.mark.asyncio
async def test_library_shell_open_in_console_triggers_handoff():
    app = _build_test_app()
    _seed_conversations(
        app,
        [
            {
                "title": "Planning Chat",
                "conversation_id": "chat-1",
                "message_count": 7,
                "updated_at": "2026-06-03T10:15:00Z",
            }
        ],
    )
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("conversation", "chat-1", "Planning Chat"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-conversations").press()
        await _wait_for_selector(screen, pilot, "#library-conversation-open-console")

        screen.query_one("#library-conversation-open-console").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "conversation"
    assert payload.source_id == "chat-1"


@pytest.mark.asyncio
async def test_library_shell_flashcards_row_renders_mode_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-flashcards").press()
        await _wait_for_selector(screen, pilot, "#library-study-handoff-detail")

        canvas = screen.query_one("#library-canvas")
        detail = screen.query_one("#library-study-handoff-detail")
        assert canvas in detail.ancestors


@pytest.mark.asyncio
async def test_library_shell_rag_open_import_export_switches_canvas_and_selection():
    app = _build_test_app()
    # Empty sources so the Search/RAG scope recovery button renders.
    _seed_conversations(app, [])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-open-import-export")

        screen.query_one("#library-rag-open-import-export").press()
        await _wait_for_selector(
            screen, pilot, "#library-import-export-workflow-title"
        )

        # The canvas now renders the Import/Export mode body, driven by the
        # shell selection rather than a bare _active_mode flip.
        canvas = screen.query_one("#library-canvas")
        title = screen.query_one("#library-import-export-workflow-title")
        assert canvas in title.ancestors
        assert not screen.query("#library-search-rag-panel")

        # ...and the rail selection marker moved to the Import/Export row.
        assert screen._library_selected_row_id == "ingest-import-export"
        row = screen.query_one("#library-row-ingest-import-export")
        assert row.has_class("library-rail-row-selected")


@pytest.mark.asyncio
async def test_library_shell_media_row_selects_canvas_without_navigating():
    """Browse ▸ Media is a canvas row now, not a screen-route shortcut.

    Re-anchors the old screen-route contract: pressing the rail row must
    render the media canvas in place (selecting it, like Conversations)
    rather than firing ``NavigateToScreen("media")``. The media SCREEN
    route now lives behind the canvas's own "Open in Media" action.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    seen = []
    host = LibraryHarness(app, seen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        await pilot.pause()

        assert seen == []
        assert screen._library_selected_row_id == "browse-media"
        assert screen.query_one("#library-media-canvas")


@pytest.mark.asyncio
async def test_library_shell_browse_media_renders_canvas_with_rows_and_preview():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-canvas")

        title = str(screen.query_one("#library-media-title").renderable)
        assert title == "Media (2)"

        filter_button = screen.query_one("#library-media-type-filter", Button)
        assert str(filter_button.label) == "type: All ▸"

        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 2
        first_label = str(rows[0].label)
        assert first_label.startswith("▸")
        # Rows sort newest-first: media-2 (10:00) before media-1 (08:00).
        assert "Product Demo Video" in first_label
        assert "video" in first_label

        preview = str(screen.query_one("#library-media-preview-lines").renderable)
        assert "Product Demo Video" in preview
        assert screen.query_one("#library-media-open")


@pytest.mark.asyncio
async def test_library_shell_media_type_filter_narrows_list():
    """Pressing the cycling filter button advances through type_options.

    The media fixture seeds two types ("audio", "video"), so
    ``type_options`` resolves to ``("All", "audio", "video")``. One press
    from the "All" default advances to "audio"; a second press advances to
    "video"; a third press wraps back around to "All".
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-canvas")

        filter_button = screen.query_one("#library-media-type-filter", Button)

        filter_button.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_type_filter == "audio"
        filter_button = screen.query_one("#library-media-type-filter", Button)
        assert str(filter_button.label) == "type: audio ▸"
        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 1
        assert "Interview Recording" in str(rows[0].label)

        status = str(screen.query_one("#library-media-status").renderable)
        assert "type: audio" in status

        filter_button.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_type_filter == "video"
        filter_button = screen.query_one("#library-media-type-filter", Button)
        assert str(filter_button.label) == "type: video ▸"
        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 1
        assert "Product Demo Video" in str(rows[0].label)

        filter_button.press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_type_filter == "All"
        filter_button = screen.query_one("#library-media-type-filter", Button)
        assert str(filter_button.label) == "type: All ▸"
        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 2


@pytest.mark.asyncio
async def test_library_shell_media_row_switches_selection():
    """Selecting a different media row updates ``_selected_media_id``.

    Re-anchored for the in-canvas viewer rebuild: pressing a media row now
    replaces the list with the full ``LibraryMediaViewer`` (no more inline
    ``#library-media-preview-lines``), so selection is asserted against the
    viewer's title/metadata instead of the old stub preview.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")

        preview_before = str(
            screen.query_one("#library-media-preview-lines").renderable
        )
        assert "Product Demo Video" in preview_before
        assert screen._selected_media_id == "media-2"

        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")

        assert screen._selected_media_id == "media-1"
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"


@pytest.mark.asyncio
async def test_library_shell_media_row_opens_full_viewer_with_content():
    """Pressing a media row switches to the in-canvas viewer.

    The viewer must show the title, Type/Author metadata lines, and the
    full stored content text (not just a 3-line preview stub), replacing
    the list (``#library-media-list`` no longer present).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")

        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")

        assert screen._library_media_view == "viewer"
        assert not screen.query("#library-media-list")

        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"

        meta = str(screen.query_one("#library-media-viewer-meta").renderable)
        assert "Type: audio" in meta
        assert "Author: Jordan Lee" in meta

        content_container = screen.query_one("#library-media-viewer-content")
        content_text = str(content_container.query_one(Static).renderable)
        assert "Full transcript: the interview recording" in content_text


@pytest.mark.asyncio
async def test_library_shell_media_use_in_chat_triggers_handoff():
    """``#library-media-use-in-chat`` stages the open media item as Console context.

    Uses the same clean ``ChatHandoffPayload`` + ``open_chat_with_handoff``
    handoff the Library conversation "Use in Console" action already uses.
    Seeds a single media item (rather than the usual two-item fixture) so
    the workspace context-handoff gate -- which is computed across *every*
    visible Library source, not just the one item under test -- isn't
    tripped up by an unlinked sibling row.
    """
    app = _build_test_app()
    media_items = _two_media_items()[:1]
    _seed_conversations(app, [], media=media_items)
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("media", "media-1", "Interview Recording"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        screen.query_one("#library-media-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-media-use-in-chat")

        screen.query_one("#library-media-use-in-chat").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "media"
    assert payload.source_id == "media-1"
    assert payload.title == "Interview Recording"
    assert "Full transcript: the interview recording" in payload.body


@pytest.mark.asyncio
async def test_library_shell_media_use_in_chat_without_open_item_notifies():
    """No handoff fires when no media item is currently loaded in the viewer.

    The "Use in Chat" button only mounts once a media item's detail has
    loaded into the viewer, so this exercises the same guard directly:
    calling the handler with ``_library_media_detail`` still at its
    freshly-mounted ``None`` must notify instead of staging anything.
    """
    app = _build_test_app()
    _seed_conversations(app, [], media=_two_media_items())
    app.open_chat_with_handoff = Mock()
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen._library_media_detail is None
        screen._open_selected_media_handoff()
        await pilot.pause()

    app.open_chat_with_handoff.assert_not_called()
    app.notify.assert_called_once()
    message = app.notify.call_args.args[0]
    assert "Open a media item" in message


@pytest.mark.asyncio
async def test_library_shell_media_back_returns_to_list():
    """``#library-media-back`` returns the media canvas to its list view."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-back")

        screen.query_one("#library-media-back").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_view == "list"
        assert screen.query_one("#library-media-list")
        assert not screen.query("#library-media-viewer-title")


@pytest.mark.asyncio
async def test_library_shell_media_rail_reentry_resets_to_list():
    """Re-entering Browse Media from the rail must show the list, not a stale viewer.

    A rail-row press is always a fresh entry into a content type. If the
    media viewer was left open on a previous visit, navigating away via
    another rail row and then pressing "Browse Media" again must land on
    the media list -- not resume the previously opened item's viewer.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")

        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")

        screen.query_one("#library-row-browse-conversations").press()
        await pilot.pause()
        await pilot.pause()

        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        await pilot.pause()

        assert screen.query_one("#library-media-list")
        assert not screen.query("#library-media-viewer-title")


@pytest.mark.asyncio
async def test_library_shell_media_viewer_shows_loading_before_detail_loads(monkeypatch):
    """The viewer shows a loading line until ``_library_media_detail`` arrives."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    monkeypatch.setattr(
        LibraryScreen, "_refresh_library_media_detail", _media_detail_never_loads
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")

        screen.query_one("#library-media-row-1").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_detail is None
        assert screen.query_one("#library-media-viewer-loading")
        assert not screen.query("#library-media-viewer-title")


async def _media_detail_never_loads(self, media_id: str) -> None:
    """Stand in for ``_refresh_library_media_detail`` that never resolves.

    Used to freeze the viewer in its loading state deterministically,
    instead of racing the real (async) detail fetch worker.
    """


@pytest.mark.asyncio
async def test_library_shell_media_open_posts_navigate_to_screen():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    seen = []
    host = LibraryHarness(app, seen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-open")

        screen.query_one("#library-media-open").press()
        await pilot.pause()
        await pilot.pause()

    assert seen[-1] == "media"


@pytest.mark.asyncio
async def test_library_shell_media_viewer_open_posts_navigate_to_screen():
    """``#library-media-open`` inside the full viewer also hands off to Media."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    seen = []
    host = LibraryHarness(app, seen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-open")

        screen.query_one("#library-media-open").press()
        await pilot.pause()
        await pilot.pause()

    assert seen[-1] == "media"


@pytest.mark.asyncio
async def test_library_shell_media_edit_shows_prefilled_form():
    """Pressing ``Edit`` swaps the metadata block for prefilled edit inputs."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit")

        screen.query_one("#library-media-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-title")

        assert screen._library_media_editing is True
        assert screen.query_one("#library-media-edit-title", Input).value == "Interview Recording"
        assert screen.query_one("#library-media-edit-author", Input).value == "Jordan Lee"
        assert screen.query_one("#library-media-edit-url", Input).value == ""
        assert (
            screen.query_one("#library-media-edit-keywords", Input).value
            == "interview, audio"
        )
        assert screen.query_one("#library-media-edit-save")
        assert screen.query_one("#library-media-edit-cancel")


@pytest.mark.asyncio
async def test_library_shell_media_edit_save_persists_and_exits_edit_mode():
    """Saving the edit form calls ``update_media_item`` and refreshes the viewer."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit")

        screen.query_one("#library-media-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-title")

        screen.query_one("#library-media-edit-title", Input).value = (
            "Interview Recording (Revised)"
        )

        screen.query_one("#library-media-edit-save").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.update_calls and not screen._library_media_editing:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Save never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert service.update_calls, "update_media_item was never called"
        call = service.update_calls[-1]
        assert call["media_id"] == "media-1"
        assert call["title"] == "Interview Recording (Revised)"
        assert call["author"] == "Jordan Lee"
        assert call["url"] == ""
        assert call["keywords"] == ["interview", "audio"]
        # `version` is NOT a supported local metadata field (see
        # LocalMediaReadingService._SUPPORTED_METADATA_FIELDS) -- sending it
        # makes every real save raise ValueError. Assert it is absent so a
        # future regression trips this test instead of silently failing in
        # production.
        assert "version" not in call

        assert screen._library_media_editing is False
        assert not screen.query("#library-media-edit-title")
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording (Revised)"


async def _open_media_edit_and_save_title(screen, pilot, new_title):
    """Open media-1's edit form, set the title, save, and wait for completion."""
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-row-1")
    screen.query_one("#library-media-row-1").press()
    await _wait_for_selector(screen, pilot, "#library-media-edit")

    screen.query_one("#library-media-edit").press()
    await _wait_for_selector(screen, pilot, "#library-media-edit-title")
    screen.query_one("#library-media-edit-title", Input).value = new_title
    screen.query_one("#library-media-edit-save").press()

    service = screen.app_instance.media_reading_scope_service
    for _ in range(150):
        if service.update_calls and not screen._library_media_editing:
            break
        await pilot.pause(0.02)
    else:
        raise AssertionError(f"Save never completed. Visible: {_visible_text(screen)}")
    await pilot.pause()


@pytest.mark.asyncio
async def test_library_shell_media_edit_save_updates_list_snapshot_cache():
    """Saving a metadata edit updates the cached list snapshot, so the media
    list shows the new title immediately instead of the pre-edit value."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_edit_and_save_title(screen, pilot, "Renamed In List")

        # The cached list snapshot record now carries the new title.
        cached = {
            screen._source_record_id(r): r
            for r in screen._local_source_records.get("media", ())
        }
        assert cached["media-1"]["title"] == "Renamed In List"

        # ...and the list view reflects it after navigating back.
        screen.query_one("#library-media-back").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        assert "Renamed In List" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_shell_media_edit_save_sanitizes_user_input():
    """User-entered edit fields are sanitized at the UI boundary, so HTML/script
    markup never reaches the persistence service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_edit_and_save_title(
            screen, pilot, "Clean<script>alert(1)</script>Title"
        )

        sent_title = screen.app_instance.media_reading_scope_service.update_calls[-1]["title"]
        assert "<script" not in sent_title.lower()
        assert "</script" not in sent_title.lower()
        # ...but the surrounding legitimate text is preserved.
        assert "Clean" in sent_title
        assert "Title" in sent_title


@pytest.mark.asyncio
async def test_library_shell_media_detail_race_discards_stale_fetch():
    """A detail fetch that completes for a no-longer-selected media id must be
    discarded instead of overwriting the currently-open viewer."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-title")
        for _ in range(150):
            detail = screen._library_media_detail
            if isinstance(detail, dict) and str(detail.get("id")) == "media-1":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("media-1 detail never loaded.")
        assert screen._selected_media_id == "media-1"

        # Simulate a slower in-flight fetch for the previously-selected media-2
        # completing now: it must not overwrite media-1's detail.
        await screen._refresh_library_media_detail("media-2")

        detail = screen._library_media_detail
        assert isinstance(detail, dict)
        assert str(detail.get("id")) == "media-1"


@pytest.mark.asyncio
async def test_library_shell_media_edit_cancel_discards():
    """Cancelling the edit form discards changes without calling the service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit")

        screen.query_one("#library-media-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-title")

        screen.query_one("#library-media-edit-title", Input).value = "Should not persist"

        screen.query_one("#library-media-edit-cancel").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_editing is False
        assert not screen.query("#library-media-edit-title")
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"

        service = app.media_reading_scope_service
        assert service.update_calls == []


@pytest.mark.asyncio
async def test_library_shell_media_delete_shows_inline_confirm_without_deleting():
    """Pressing ``Delete`` shows the inline confirm affordance, not an immediate delete."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete")

        screen.query_one("#library-media-delete").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")

        assert screen._library_media_confirming_delete is True
        assert screen.query_one("#library-media-delete-cancel")

        service = app.media_reading_scope_service
        assert service.delete_calls == []
        assert screen._library_media_view == "viewer"


@pytest.mark.asyncio
async def test_library_shell_media_delete_confirm_removes_item_and_returns_to_list():
    """Confirming the delete trashes the item and drops it from the list view."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete")

        screen.query_one("#library-media-delete").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")

        screen.query_one("#library-media-delete-confirm").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.delete_calls and screen._library_media_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Delete never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert service.delete_calls, "delete_media_item was never called"
        assert service.delete_calls[-1]["media_id"] == "media-1"
        assert screen._library_media_confirming_delete is False
        assert screen._library_media_view == "list"
        assert not screen.query("#library-media-viewer")
        # media-1 ("Interview Recording") is deleted; only media-2 ("Product
        # Demo Video") remains, re-indexed to row-0 by the sorted rebuild.
        assert not any(
            "Interview Recording" in str(getattr(button, "label", ""))
            for button in screen.query(".library-media-row")
        )
        assert screen.query_one("#library-media-row-0")


@pytest.mark.asyncio
async def test_library_shell_media_delete_cancel_leaves_item_intact():
    """Cancelling the inline confirm discards it without calling the service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete")

        screen.query_one("#library-media-delete").press()
        await _wait_for_selector(screen, pilot, "#library-media-delete-confirm")

        screen.query_one("#library-media-delete-cancel").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_confirming_delete is False
        assert not screen.query("#library-media-delete-confirm")
        assert screen.query_one("#library-media-delete")
        title = str(screen.query_one("#library-media-viewer-title").renderable)
        assert title == "Interview Recording"

        service = app.media_reading_scope_service
        assert service.delete_calls == []


@pytest.mark.asyncio
async def test_library_shell_media_viewer_renders_seeded_highlight():
    """A seeded highlight's quote renders in the viewer's highlights section."""
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        media=_two_media_items(),
        highlights=[("media-1", "Important sentence", "Check this", "yellow")],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-highlight-0")

        visible = _visible_text(screen)
        assert "Important sentence" in visible
        assert "Check this" in visible
        # A renderable color is conveyed by a tinted "●" swatch on the quote,
        # not by the literal word "yellow".
        quote = screen.query_one("#library-media-highlight-0").renderable
        assert "●" in quote.plain
        assert any("yellow" in str(span.style).lower() for span in quote.spans)
        assert screen.query_one("#library-media-highlight-delete-0")


@pytest.mark.asyncio
async def test_library_shell_media_highlight_non_rich_color_does_not_crash_render():
    """A Textual-valid but Rich-invalid color (e.g. 'transparent') must not crash.

    The swatch color is consumed as a Rich style; validating with Textual's
    (superset) parser let 'transparent'/'hsl(...)'/'ansi_*' through and then
    crashed at render (rich.errors.MissingStyle). Such colors must fall back
    to plain text instead.
    """
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        media=_two_media_items(),
        highlights=[("media-1", "Fragile quote", None, "transparent")],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        # Reaching this selector means the viewer rendered without raising.
        await _wait_for_selector(screen, pilot, "#library-media-highlight-0")

        quote = screen.query_one("#library-media-highlight-0").renderable
        # No swatch marker (color not renderable); it survives as plain text.
        assert "●" not in quote.plain
        assert "transparent" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_shell_media_highlight_add_creates_and_renders_new_highlight():
    """Filling the quote input and pressing Add calls ``create_highlight`` and renders it."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-highlight-add-collapsible")

        assert not screen.query(".library-media-highlight-delete")

        # The add-highlight form starts collapsed; expand it before typing
        # into its inputs, mirroring how a real user would reveal the form.
        collapsible = screen.query_one(
            "#library-media-highlight-add-collapsible", Collapsible
        )
        assert collapsible.collapsed
        collapsible.collapsed = False
        await pilot.pause()

        screen.query_one("#library-media-highlight-quote", Input).value = "New highlight quote"
        screen.query_one("#library-media-highlight-add").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.create_highlight_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Add never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()
        await pilot.pause()

        assert service.create_highlight_calls[-1]["item_id"] == "media-1"
        assert service.create_highlight_calls[-1]["quote"] == "New highlight quote"
        await _wait_for_selector(screen, pilot, "#library-media-highlight-0")
        assert "New highlight quote" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_shell_media_highlight_delete_removes_it():
    """Pressing a highlight's delete button calls ``delete_highlight`` and removes the row."""
    app = _build_test_app()
    _seed_conversations(
        app,
        _two_conversations(),
        media=_two_media_items(),
        highlights=[("media-1", "Doomed highlight", None, None)],
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-highlight-delete-0")

        screen.query_one("#library-media-highlight-delete-0").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.delete_highlight_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Delete never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()
        await pilot.pause()

        assert service.delete_highlight_calls[-1]["highlight_id"] == "1"
        assert "Doomed highlight" not in _visible_text(screen)
        assert not screen.query(".library-media-highlight-delete")


@pytest.mark.asyncio
async def test_library_shell_media_read_later_saves_and_flips_button_label():
    """Pressing "Read it later" calls save_to_read_it_later and flips the label."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-read-later")

        assert str(screen.query_one("#library-media-read-later", Button).label) == "Read it later"

        screen.query_one("#library-media-read-later").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.read_it_later_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Toggle never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()
        await pilot.pause()

        assert service.read_it_later_calls[-1] == {"action": "save", "media_id": "media-1"}
        assert screen._library_media_detail["is_read_it_later"] is True
        assert (
            str(screen.query_one("#library-media-read-later", Button).label)
            == "Remove from read-it-later"
        )


@pytest.mark.asyncio
async def test_library_shell_media_read_later_removes_when_already_saved():
    """Pressing the toggle on an already-saved item calls remove_from_read_it_later."""
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["is_read_it_later"] = True
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-read-later")

        assert (
            str(screen.query_one("#library-media-read-later", Button).label)
            == "Remove from read-it-later"
        )

        screen.query_one("#library-media-read-later").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.read_it_later_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Toggle never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()
        await pilot.pause()

        assert service.read_it_later_calls[-1] == {"action": "remove", "media_id": "media-1"}
        assert "is_read_it_later" not in screen._library_media_detail
        assert str(screen.query_one("#library-media-read-later", Button).label) == "Read it later"


@pytest.mark.asyncio
async def test_library_shell_media_viewer_shows_analysis_from_latest_version():
    """Analysis text from the newest DocumentVersions row renders in the viewer.

    Local media details never carry top-level analysis_content (it lives on
    DocumentVersions only) -- this proves the viewer surfaces it from the
    ``versions`` list returned by ``get_media_item``.
    """
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = [
        {"version_number": 2, "analysis_content": "Roadmap analysis"},
        {"version_number": 1, "analysis_content": None},
    ]
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-viewer-analysis-text")

        analysis_text = str(screen.query_one("#library-media-viewer-analysis-text").renderable)
        assert analysis_text == "Roadmap analysis"


@pytest.mark.asyncio
async def test_library_shell_media_analysis_edit_shows_prefilled_textarea():
    """Pressing "Edit analysis" swaps the analysis text for a prefilled TextArea."""
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = [{"version_number": 1, "analysis_content": "Existing analysis"}]
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")

        screen.query_one("#library-media-analysis-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")

        assert screen._library_media_editing_analysis is True
        text_area = screen.query_one("#library-media-analysis-edit-text", TextArea)
        assert text_area.text == "Existing analysis"
        assert screen.query_one("#library-media-analysis-save")
        assert screen.query_one("#library-media-analysis-cancel")


@pytest.mark.asyncio
async def test_library_shell_media_analysis_save_persists_and_exits_edit_mode():
    """Saving the analysis edit form calls save_analysis_version and refreshes the viewer."""
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = [{"version_number": 1, "analysis_content": "Existing analysis"}]
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")

        screen.query_one("#library-media-analysis-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")

        screen.query_one("#library-media-analysis-edit-text", TextArea).text = (
            "Revised analysis"
        )

        screen.query_one("#library-media-analysis-save").press()

        service = app.media_reading_scope_service
        for _ in range(150):
            if service.analysis_calls and not screen._library_media_editing_analysis:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Save never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert service.analysis_calls, "save_analysis_version was never called"
        call = service.analysis_calls[-1]
        assert call["media_id"] == "media-1"
        assert call["analysis_content"] == "Revised analysis"
        assert call["content"] == "Full transcript: the interview recording covers the quarterly roadmap."

        assert screen._library_media_editing_analysis is False
        assert not screen.query("#library-media-analysis-edit-text")
        analysis_text = str(screen.query_one("#library-media-viewer-analysis-text").renderable)
        assert analysis_text == "Revised analysis"


@pytest.mark.asyncio
async def test_library_shell_media_analysis_cancel_discards():
    """Cancelling the analysis edit form discards changes without calling the service."""
    app = _build_test_app()
    media_items = _two_media_items()
    media_items[0]["versions"] = [{"version_number": 1, "analysis_content": "Existing analysis"}]
    _seed_conversations(app, _two_conversations(), media=media_items)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-1")
        screen.query_one("#library-media-row-1").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")

        screen.query_one("#library-media-analysis-edit").press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")

        screen.query_one("#library-media-analysis-edit-text", TextArea).text = "Should not persist"

        screen.query_one("#library-media-analysis-cancel").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_editing_analysis is False
        assert not screen.query("#library-media-analysis-edit-text")
        analysis_text = str(screen.query_one("#library-media-viewer-analysis-text").renderable)
        assert analysis_text == "Existing analysis"

        service = app.media_reading_scope_service
        assert service.analysis_calls == []


def _media_item_with_multiline_content():
    """A media item whose content has two lines containing "budget"."""
    items = _two_media_items()
    items[0]["content"] = "\n".join(
        [
            "Intro paragraph about the roadmap.",
            "The budget forecast is discussed here.",
            "Another neutral line.",
            "Second budget line appears here too.",
        ]
    )
    return items


def _media_item_with_two_hits_on_one_line():
    """A media item whose only matching line contains "budget" twice."""
    items = _two_media_items()
    items[0]["content"] = "The budget and the budget again on one single line."
    return items


async def _open_media_viewer(screen, pilot):
    """Navigate to the media list and open the first row's viewer."""
    screen.query_one("#library-row-browse-media").press()
    await _wait_for_selector(screen, pilot, "#library-media-row-1")
    screen.query_one("#library-media-row-1").press()
    await _wait_for_selector(screen, pilot, "#library-media-content-search")


async def _submit_content_search_query(screen, pilot, query):
    """Type ``query`` into the open viewer's content search box and press Enter."""
    search_input = screen.query_one("#library-media-content-search", Input)
    search_input.value = query
    search_input.focus()
    await pilot.pause()
    await pilot.press("enter")
    await pilot.pause()
    await pilot.pause()


async def _open_media_viewer_and_submit_content_search(screen, pilot, query):
    """Open the first media row's viewer and submit a content-search query."""
    await _open_media_viewer(screen, pilot)
    await _submit_content_search_query(screen, pilot, query)


@pytest.mark.asyncio
async def test_library_shell_media_content_search_no_query_hides_status_and_nav():
    """With no active search, only the search box renders -- no orphaned status/prev/next."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer(screen, pilot)

        assert screen.query_one("#library-media-content-search")
        assert not screen.query("#library-media-content-search-status")
        assert not screen.query("#library-media-content-search-prev")
        assert not screen.query("#library-media-content-search-next")


@pytest.mark.asyncio
async def test_library_shell_media_content_search_shows_match_count():
    """Submitting a query shows the match count and starts at the first match."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")

        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 1 of 2 matches"
        assert screen._library_media_content_query == "budget"
        assert screen._library_media_content_match_index == 0


@pytest.mark.asyncio
async def test_library_shell_media_content_search_highlights_matches_in_body():
    """While searching, each query occurrence in the content body is styled."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")

        content = screen.query_one("#library-media-viewer-content-text").renderable
        # Rendered as a Rich Text with a styled span over each "budget".
        highlighted = [
            content.plain[span.start : span.end]
            for span in content.spans
            if str(span.style)
        ]
        assert highlighted.count("budget") == 2
        assert all(part.lower() == "budget" for part in highlighted)


@pytest.mark.asyncio
async def test_library_shell_media_content_search_one_mark_per_matching_line():
    """A line with two hits is one match and gets one mark (count == visible marks)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_two_hits_on_one_line())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")

        # One matching line -> "Match 1 of 1", even though "budget" appears twice.
        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 1 of 1 matches"
        # ...and exactly one styled mark in the body, so count == visible marks.
        content = screen.query_one("#library-media-viewer-content-text").renderable
        marks = [span for span in content.spans if str(span.style)]
        assert len(marks) == 1
        assert content.plain[marks[0].start : marks[0].end] == "budget"


@pytest.mark.asyncio
async def test_library_shell_media_content_search_no_matches_shows_status():
    """A query with no hits in the content shows a "No matches" status."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "nonexistent-term")

        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "No matches"


@pytest.mark.asyncio
async def test_library_shell_media_content_search_empty_query_hides_status_and_nav():
    """Submitting a blank query clears the query and hides the status/prev/next row."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")
        assert screen.query_one("#library-media-content-search-status")

        await _submit_content_search_query(screen, pilot, "")

        # With no active query, the status line and prev/next toolbar are not
        # rendered at all -- they aren't just blank, they're gone -- so the
        # orphaned nav doesn't linger under the search box.
        assert not screen.query("#library-media-content-search-status")
        assert not screen.query("#library-media-content-search-prev")
        assert not screen.query("#library-media-content-search-next")
        assert screen._library_media_content_query == ""


@pytest.mark.asyncio
async def test_library_shell_media_content_search_next_prev_advances_match_index():
    """Next/Prev cycle the match index (wrapping) and update the status line."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")
        assert screen._library_media_content_match_index == 0

        screen.query_one("#library-media-content-search-next").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_content_match_index == 1
        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 2 of 2 matches"

        # Next wraps back around to the first match.
        screen.query_one("#library-media-content-search-next").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_content_match_index == 0
        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 1 of 2 matches"

        # Prev wraps backwards to the last match.
        screen.query_one("#library-media-content-search-prev").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_content_match_index == 1
        status = str(screen.query_one("#library-media-content-search-status").renderable)
        assert status == "Match 2 of 2 matches"


@pytest.mark.asyncio
async def test_library_shell_media_content_search_resets_on_back():
    """Returning to the media list clears the in-content search state."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), media=_media_item_with_multiline_content())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        await _open_media_viewer_and_submit_content_search(screen, pilot, "budget")
        assert screen._library_media_content_query == "budget"

        screen.query_one("#library-media-back").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_media_content_query == ""
        assert screen._library_media_content_match_index == 0


@pytest.mark.asyncio
async def test_library_shell_media_canvas_shows_loading_before_snapshot_loads(monkeypatch):
    """Mirrors the conversations loading-gate contract for the media canvas."""
    app = _build_test_app()
    _seed_conversations(app, [])

    monkeypatch.setattr(LibraryScreen, "_refresh_local_source_snapshot", _never_loads)

    screen = LibraryScreen(app)
    screen._library_selected_row_id = "browse-media"
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await pilot.pause()
        await pilot.pause()

        assert active_screen._library_loaded is False
        assert active_screen.query_one("#library-canvas-loading")
        assert not active_screen.query("#library-media-canvas")


@pytest.mark.asyncio
async def test_library_shell_search_filters_conversations_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        search_input = screen.query_one("#library-search-input")
        search_input.value = "quarterly"
        search_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-conversations-status")

        status = str(screen.query_one("#library-conversations-status").renderable)
        assert status == "1 match for 'quarterly'"
        rows = list(screen.query(".library-conversation-row"))
        assert len(rows) == 1


@pytest.mark.asyncio
async def test_library_shell_search_retains_value_after_submit():
    """Submitting a search recomposes the shell; the box must keep the query.

    The submit handler rebuilds the whole screen (``refresh(recompose=True)``),
    which remounts a brand-new ``#library-search-input``. Regression guard:
    that new input must be seeded with the active query instead of showing
    empty text while the filter is silently active.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        search_input = screen.query_one("#library-search-input")
        search_input.value = "quarterly"
        search_input.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-conversations-status")

        recomposed_input = screen.query_one("#library-search-input")
        assert recomposed_input.value == "quarterly"
        assert recomposed_input.has_focus


def _never_loads(self) -> None:
    """Stand in for ``_refresh_local_source_snapshot`` that never resolves.

    Used to freeze ``LibraryScreen`` in its pre-load state deterministically,
    instead of racing the real snapshot worker to assert on a narrow timing
    window.
    """
    return None


@pytest.mark.asyncio
async def test_library_shell_shows_loading_state_before_snapshot_loads(monkeypatch):
    """Before the local source snapshot loads, the canvas must show a loading
    indicator instead of the false "no conversations" empty state, and the
    rail must not claim a known zero count for Conversations.
    """
    app = _build_test_app()
    _seed_conversations(app, [])

    monkeypatch.setattr(LibraryScreen, "_refresh_local_source_snapshot", _never_loads)

    screen = LibraryScreen(app)
    screen.apply_navigation_context({"mode": "conversations"})
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await pilot.pause()
        await pilot.pause()

        assert active_screen._library_loaded is False
        assert active_screen.query_one("#library-canvas-loading")
        assert not active_screen.query("#library-conversations-canvas")

        visible = _visible_text(active_screen)
        assert "Conversations (0)" not in visible


@pytest.mark.asyncio
async def test_library_shell_shows_lookup_error_in_canvas(monkeypatch):
    """A local-source lookup error must surface in the canvas, not only in
    the (possibly collapsed) Details disclosure.
    """
    app = _build_test_app()
    _seed_conversations(app, [])

    monkeypatch.setattr(LibraryScreen, "_refresh_local_source_snapshot", _never_loads)

    screen = LibraryScreen(app)
    screen.apply_navigation_context({"mode": "conversations"})
    screen._library_lookup_error = "Library sources are unavailable right now."
    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        active_screen = _active_library_screen(host)
        await pilot.pause()
        await pilot.pause()

        error_static = active_screen.query_one("#library-canvas-error")
        assert "unavailable" in str(error_static.renderable).lower()
        assert not active_screen.query("#library-canvas-loading")


@pytest.mark.asyncio
async def test_library_shell_details_toggle_persists():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert (
            screen.query_one("#library-rail-section-body-details").styles.display
            == "none"
        )
        toggle = screen.query_one("#console-rail-section-toggle-library-details")
        toggle.press()
        await pilot.pause()
        await pilot.pause()

        assert (
            screen.query_one("#library-rail-section-body-details").styles.display
            != "none"
        )

    sections = (
        app.app_config.get("library", {}).get("rail_state", {}).get("sections", {})
    )
    assert sections.get("details_open") is True


@pytest.mark.asyncio
async def test_library_shell_collections_row_loads_seeded_records():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_collections_service = StaticLibraryCollectionsService(
        [
            {
                "collection_id": "collection-1",
                "name": "Launch Evidence",
                "description": "Sources for release review.",
                "item_count": 3,
                "source_authority": "local",
                "sync_status": "local-only",
                "updated_at": "2026-06-09T12:00:00Z",
            }
        ]
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Entering Collections via the rail row must load the snapshot with no
        # prior create/rename action; the seeded record renders in the canvas.
        screen.query_one("#library-row-browse-collections").press()
        select_button = await _wait_for_selector(
            screen, pilot, "#library-collection-select-0"
        )

        canvas = screen.query_one("#library-canvas")
        assert canvas in select_button.ancestors
        assert "Launch Evidence" in str(select_button.label)


@pytest.mark.asyncio
async def test_library_shell_collections_deeplink_loads_before_mount():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_collections_service = StaticLibraryCollectionsService(
        [
            {
                "collection_id": "collection-1",
                "name": "Launch Evidence",
                "description": "Sources for release review.",
                "item_count": 3,
                "source_authority": "local",
                "sync_status": "local-only",
                "updated_at": "2026-06-09T12:00:00Z",
            }
        ]
    )
    screen = LibraryScreen(app)

    # Mirrors the real app.py ordering: handle_screen_navigation calls
    # apply_navigation_context BEFORE switch_screen mounts the destination
    # screen, so the screen is never mounted at this point.
    assert screen.is_mounted is False
    screen.apply_navigation_context({"mode": "collections"})

    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        # Deep-linking straight into Collections (no prior rail-row press)
        # must still load the snapshot; the seeded record renders in the
        # canvas without requiring any additional user interaction.
        select_button = await _wait_for_selector(
            screen, pilot, "#library-collection-select-0"
        )

        canvas = screen.query_one("#library-canvas")
        assert canvas in select_button.ancestors
        assert "Launch Evidence" in str(select_button.label)


@pytest.mark.asyncio
async def test_library_shell_notes_create_deeplink_lands_on_create_view():
    """The retired Notes tab's "new note" deep link now re-points into
    Library: a ``notes_create`` navigation context must land the shell on
    the in-canvas Create > New note view, mirroring how pressing the
    "New note" rail row does (``LIBRARY_ROW_CREATE_NOTE`` / canvas kind
    ``notes-create``).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    screen = LibraryScreen(app)

    # Mirrors the real app.py ordering: handle_screen_navigation calls
    # apply_navigation_context BEFORE switch_screen mounts the destination
    # screen (see test_library_shell_collections_deeplink_loads_before_mount).
    assert screen.is_mounted is False
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_NOTES_CREATE: True})

    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        assert screen._library_selected_row_id == LIBRARY_ROW_CREATE_NOTE
        assert screen.query_one("#library-notes-create-blank")


@pytest.mark.asyncio
async def test_library_shell_note_id_deeplink_opens_note_editor():
    """The retired Notes tab's chat-sidebar deep link now re-points into
    Library: a ``note_id`` navigation context must open that note's
    in-canvas editor directly, without requiring a prior rail-row press.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    screen = LibraryScreen(app)

    assert screen.is_mounted is False
    screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_NOTE_ID: "n-1"})

    host = LibraryHarness(app, screen=screen)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-note-title")

        assert screen._library_selected_row_id == LIBRARY_ROW_BROWSE_NOTES
        assert screen._selected_note_id == "n-1"
        title = screen.query_one("#library-note-title", Input)
        assert title.value == "Q3 retro"


@pytest.mark.asyncio
async def test_library_shell_workspaces_body_lives_under_details():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        toggle = screen.query_one("#console-rail-section-toggle-library-details")
        toggle.press()
        await pilot.pause()
        await pilot.pause()

        details_body = screen.query_one("#library-rail-section-body-details")
        create_button = screen.query_one("#library-create-local-workspace")
        assert details_body in create_button.ancestors


def test_generated_stylesheet_includes_library_shell_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        "#library-shell-grid",
        "#library-header-line",
        "#library-rail",
        "#library-canvas",
        ".library-rail-empty-copy",
        ".library-rail-row",
        ".library-rail-row-selected",
        ".library-conversation-row",
        ".library-conversation-row-selected",
        ".library-canvas-action",
        "#library-search-input",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
    for stale in ("#library-mode-bar", "#library-contract-grid"):
        assert stale not in component_css, stale


def test_generated_stylesheet_includes_library_media_rules():
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    component_css = (root / "components" / "_agentic_terminal.tcss").read_text()
    generated_css = (root / "tldw_cli_modular.tcss").read_text()
    for selector in (
        "#library-media-title",
        ".library-media-row",
        ".library-media-row-selected",
        "#library-media-type-filter",
    ):
        assert selector in component_css, selector
        assert selector in generated_css, selector
    assert "#library-media-header" not in component_css


def _css_rule_body(css_text: str, selector: str) -> str:
    """Return the declaration body of the first ``selector { ... }`` rule.

    Args:
        css_text: Full stylesheet text to search.
        selector: A literal selector (e.g. ``"#library-rail"``) that opens a
            rule, possibly as part of a comma-separated selector list.

    Returns:
        The text between the rule's opening ``{`` and its matching ``}``.

    Raises:
        AssertionError: If the selector never opens a rule block.
    """
    pattern = re.compile(re.escape(selector) + r"\s*[,{]")
    match = pattern.search(css_text)
    assert match, f"{selector} does not open a CSS rule"
    brace_start = css_text.index("{", match.start())
    brace_end = css_text.index("}", brace_start)
    return css_text[brace_start + 1 : brace_end]


def test_library_rail_css_scrolls_vertically_with_scrollbar_styling():
    """The rail must scroll so Details content past the viewport stays reachable.

    Regression for the live-QA overflow: the rail is a plain ``Vertical``
    (Textual default ``overflow: hidden hidden``), so once every section is
    open the Details body and its workspace action buttons render past the
    pane's bottom and are neither visible nor reachable. Scoping to the
    literal ``#library-rail`` rule (not just "selector appears somewhere in
    the file") keeps this from passing on an unrelated rule that happens to
    mention the same properties elsewhere.
    """
    root = Path(__file__).resolve().parents[2] / "tldw_chatbook" / "css"
    for css_path in (
        root / "components" / "_agentic_terminal.tcss",
        root / "tldw_cli_modular.tcss",
    ):
        body = _css_rule_body(css_path.read_text(), "#library-rail")
        assert "overflow-y: auto" in body, css_path
        assert "scrollbar-background: $ds-surface-panel" in body, css_path
        assert "scrollbar-color: $ds-grid-line" in body, css_path


@pytest.mark.asyncio
async def test_library_shell_rail_scrolls_to_reveal_workspace_actions():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        for section in ("browse", "create", "ingest", "details"):
            body = screen.query_one(f"#library-rail-section-body-{section}")
            if body.styles.display == "none":
                screen.query_one(f"#console-rail-section-toggle-library-{section}").press()
                await pilot.pause()
                await pilot.pause()

        rail = screen.query_one("#library-rail")
        assert str(rail.styles.overflow_y) == "auto"

        rail.scroll_end(animate=False)
        await pilot.pause()
        await pilot.pause()

        for selector in ("#library-create-local-workspace", "#library-use-in-console"):
            button = screen.query_one(selector)
            assert rail.region.y <= button.region.y < rail.region.y + rail.region.height, (
                f"{selector} region {button.region} is not within the scrolled rail "
                f"viewport {rail.region}"
            )


@pytest.mark.asyncio
async def test_library_shell_details_body_has_no_duplicate_collapse_control():
    """The moved Workspaces body must not carry its own collapse affordance.

    The Details section already owns exactly one collapse/expand toggle (the
    rail's ``ConsoleRailSectionHeader``, a sibling of the body, tooltipped
    "Collapse Details"/"Expand Details"). Nothing inside the body itself
    should duplicate that affordance.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        toggle = screen.query_one("#console-rail-section-toggle-library-details")
        toggle.press()
        await pilot.pause()
        await pilot.pause()

        details_body = screen.query_one("#library-rail-section-body-details")
        for button in details_body.query(Button):
            label = str(button.label)
            tooltip = button.tooltip or ""
            assert "Collapse Details" not in label
            assert "Collapse Details" not in tooltip


@pytest.mark.asyncio
async def test_library_shell_media_use_in_chat_body_truncated_for_long_content():
    """``body_truncated`` is True when media content exceeds 500 chars.

    The handoff excerpt is capped at LIBRARY_MEDIA_HANDOFF_EXCERPT_CHARS (500),
    so long content must set body_truncated=True. This test seeds media with
    600 chars and asserts both the flag and that the excerpt is bounded.
    """
    app = _build_test_app()
    media_items = _two_media_items()[:1]
    # Seed with content > 500 chars: use a marker string repeated to exceed 500.
    long_content = "This is long content. " * 35  # ~770 chars
    media_items[0]["content"] = long_content
    _seed_conversations(app, [], media=media_items)
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("media", "media-1", "Interview Recording"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        screen.query_one("#library-media-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-media-use-in-chat")

        screen.query_one("#library-media-use-in-chat").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "media"
    assert payload.source_id == "media-1"
    assert payload.title == "Interview Recording"
    assert payload.body_truncated is True
    # The payload.body includes metadata headers plus the excerpt.
    # Verify the excerpt is bounded: a marker string only in the first
    # 500 chars should appear, but content after position 500 should not.
    assert "This is long content." in payload.body
    # Verify truncation: content after char 500 should not be present.
    # The full content string repeated would produce a pattern that continues,
    # so if truncated, it will end mid-word or mid-pattern.
    assert "Content excerpt:" in payload.body
    # Count the actual content after "Content excerpt:" header
    excerpt_start = payload.body.find("Content excerpt:")
    content_part = payload.body[excerpt_start:]
    # The excerpt part should be much shorter than the original content
    assert len(content_part) < len(long_content)


@pytest.mark.asyncio
async def test_library_shell_media_use_in_chat_body_not_truncated_for_short_content():
    """``body_truncated`` is False when media content is <= 500 chars.

    Short content does not set the truncated flag. This test seeds media with
    300 chars and asserts body_truncated=False and that the full content
    is included in the excerpt.
    """
    app = _build_test_app()
    media_items = _two_media_items()[:1]
    # Seed with content < 500 chars: a 300-char realistic string
    short_content = (
        "This is a medium-length transcript excerpt that stays well under "
        "the 500-character limit. It contains enough text to be substantial "
        "but short enough to pass through without truncation. This demonstrates "
        "that the body_truncated flag correctly handles content that fits "
        "entirely within the handoff excerpt boundary."
    )
    media_items[0]["content"] = short_content
    _seed_conversations(app, [], media=media_items)
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("media", "media-1", "Interview Recording"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")
        screen.query_one("#library-media-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-media-use-in-chat")

        screen.query_one("#library-media-use-in-chat").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "media"
    assert payload.source_id == "media-1"
    assert payload.title == "Interview Recording"
    assert payload.body_truncated is False
    # Full content should be present (not truncated) in the body payload.
    assert short_content in payload.body
    # Verify the structure includes the excerpt section.
    assert "Content excerpt:" in payload.body


def _two_notes():
    return [
        {"id": "n-1", "title": "Q3 retro", "content": "alpha budget line",
         "last_modified": "2026-07-07T11:57:00+00:00", "version": 2},
        {"id": "n-2", "title": "Reading list", "content": "bravo",
         "last_modified": "2026-07-06T12:00:00+00:00", "version": 1},
    ]


class _CountSeamLibraryNotesScopeService(StaticLibraryNotesListScopeService):
    """Local notes fake mirroring the real production shape: ``list_notes``
    returns a bare list with no total (like ``NotesInteropService.list_notes``),
    but ``count_notes`` (the new seam under test) gives the exact total.

    Used to prove the Library screen's rail badge renders an exact count
    even though the paginated ``list_notes`` response alone cannot supply
    one -- unlike ``StaticLibraryNotesScopeService``, whose ``list_notes``
    already carries a ``pagination.total`` and so cannot distinguish
    "badge is exact because of ``count_notes``" from "badge is exact
    because ``list_notes`` said so".
    """

    def __init__(self, notes):
        super().__init__(notes)
        self.count_calls = []

    async def count_notes(self, *, scope, user_id=None, **kwargs):
        self.count_calls.append({"scope": scope, "user_id": user_id, **kwargs})
        return len(self.notes)


@pytest.mark.asyncio
async def test_library_shell_notes_rail_badge_shows_exact_count_via_count_seam():
    """The rail badge renders an exact ``(2)`` -- no "+" sample-cap suffix
    -- once ``count_notes`` is wired into the Library screen's local-source
    snapshot fetch, even though the underlying ``list_notes`` response
    carries no total of its own (the real production shape)."""
    app = _build_test_app()
    app.notes_scope_service = _CountSeamLibraryNotesScopeService(_two_notes())
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        rail_label = str(screen.query_one("#library-row-browse-notes").label)
        assert "(2)" in rail_label
        assert "(2+)" not in rail_label
    assert app.notes_scope_service.count_calls


@pytest.mark.asyncio
async def test_library_shell_notes_rail_badge_degrades_without_count_seam():
    """When the local notes service exposes no ``count_notes`` seam (the
    plain-list fake never defines one, mirroring a backend that hasn't
    adopted the new seam), the rail badge keeps today's "showing up to N"
    sample-cap contract (``(N+)``) instead of claiming an exact total it
    cannot verify -- and the snapshot fetch does not error out."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService(_two_notes())
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        rail_label = str(screen.query_one("#library-row-browse-notes").label)
        assert "(2+)" in rail_label


@pytest.mark.asyncio
async def test_library_shell_notes_row_opens_notes_list_canvas():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        header = str(screen.query_one("#library-notes-header").renderable)
        assert header == "Notes (2)"
        assert screen.query_one("#library-notes-filter")
        assert screen.query_one("#library-notes-sort")


@pytest.mark.asyncio
async def test_library_shell_notes_sort_button_cycles():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-sort")
        assert screen._library_notes_sort == "newest"
        screen.query_one("#library-notes-sort").press()
        await pilot.pause()
        assert screen._library_notes_sort == "oldest"


@pytest.mark.asyncio
async def test_library_shell_notes_filter_queries_search_seam():
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")
        box = screen.query_one("#library-notes-filter", Input)
        box.value = "retro"
        box.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause(); await pilot.pause()
        service = app.notes_scope_service
        assert service.search_calls[-1]["query"] == "retro"
        assert screen._library_notes_filter == "retro"


class _GatedSearchLibraryNotesScopeService(StaticLibraryNotesScopeService):
    """A ``search_notes`` that blocks until the test releases it, so a
    filter test can submit, observe the seam call start, mutate screen
    state while that call is still in flight, and only then let the stale
    response resolve on demand.

    Uses a ``threading.Event`` rather than an ``asyncio.Event``:
    ``_run_library_service_call`` runs this seam with
    ``isolate_in_worker=True``, which executes it in a *different* thread
    under a freshly-created event loop (``asyncio.run`` inside
    ``asyncio.to_thread``) -- an ``asyncio.Event`` set from the test's own
    thread/loop would not safely wake a waiter parked on that other loop.
    """

    def __init__(self, notes):
        super().__init__(notes)
        self.release_event = threading.Event()

    async def search_notes(self, **kwargs):
        self.search_calls.append(kwargs)
        await asyncio.to_thread(self.release_event.wait)
        return await super().search_notes(
            scope=kwargs.get("scope"),
            query=kwargs.get("query"),
            limit=kwargs.get("limit"),
            user_id=kwargs.get("user_id"),
            offset=kwargs.get("offset", 0),
        )


@pytest.mark.asyncio
async def test_library_shell_notes_filter_clears_before_stale_response_lands():
    """Clearing the filter while a slower ``search_notes`` call is still in
    flight must not let that stale response overwrite the cleared filter
    state once it resolves -- the same stale-result discipline the note
    detail fetch and save already apply to their own out-of-order results.
    """
    app = _build_test_app()
    service = _GatedSearchLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")

        box = screen.query_one("#library-notes-filter", Input)
        box.value = "retro"
        box.focus()
        await pilot.pause()
        await pilot.press("enter")

        for _ in range(150):
            if service.search_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Filter submit never called search_notes.")

        assert screen._library_notes_filter == "retro"

        # Clear the filter while the "retro" search is still gated in flight.
        clear_box = screen.query_one("#library-notes-filter", Input)
        clear_box.value = ""
        clear_box.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()

        assert screen._library_notes_filter == ""
        assert screen._library_notes_filter_records is None

        # Now release the stale "retro" response.
        service.release_event.set()
        for _ in range(20):
            await pilot.pause(0.02)

        assert screen._library_notes_filter_records is None, (
            "The stale in-flight filter response overwrote the cleared filter state."
        )
        header = str(screen.query_one("#library-notes-header").renderable)
        assert header == "Notes (2)"


@pytest.mark.asyncio
async def test_library_shell_notes_row_opens_editor_with_detail():
    """Pressing a note row fetches the full detail and renders the in-canvas
    editor prefilled with the note's title, body, and version."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        title = screen.query_one("#library-note-title", Input)
        assert title.value == "Q3 retro"
        body = screen.query_one("#library-note-body", TextArea)
        assert body.text == "alpha budget line"
        meta = str(screen.query_one("#library-note-meta").renderable)
        assert "v2" in meta
        assert screen._library_notes_view == "editor"
        assert screen._selected_note_id == "n-1"


class StaticLibraryNotesKeywordsService:
    """Fake ``app.notes_service`` stand-in for the keywords-enrichment seam
    ``_refresh_library_note_detail`` uses (mirroring
    ``NotesInteropService.get_keywords_for_note``'s real
    ``(user_id, note_id) -> list[{"keyword": ...}]`` shape -- see
    ``ChaChaNotes_DB.get_keywords_for_note``, whose rows always carry a
    ``keyword`` column).
    """

    def __init__(self, keywords_by_note_id):
        self.keywords_by_note_id = dict(keywords_by_note_id)
        self.calls = []

    def get_keywords_for_note(self, user_id, note_id):
        self.calls.append({"user_id": user_id, "note_id": note_id})
        return [
            {"id": index, "keyword": keyword}
            for index, keyword in enumerate(
                self.keywords_by_note_id.get(str(note_id), []), start=1
            )
        ]


@pytest.mark.asyncio
async def test_library_shell_note_editor_shows_enriched_keywords():
    """``notes_scope_service.get_note_detail``'s local-scope shape is the
    raw ``notes`` table row and never carries a ``keywords`` field (see
    ``NotesInteropService.get_note_by_id``), so the in-canvas editor's
    Keywords input would otherwise always render empty even for a note
    with persisted keywords. ``_refresh_library_note_detail`` must enrich
    the fetched detail via the ``app.notes_service.get_keywords_for_note``
    seam before building editor state.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notes_service = StaticLibraryNotesKeywordsService({"n-1": ["kw1", "kw2"]})
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        keywords_input = screen.query_one("#library-note-keywords", Input)
        for _ in range(150):
            if keywords_input.value == "kw1, kw2":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Keywords input never showed enriched keywords: {keywords_input.value!r}"
            )

        assert app.notes_service.calls
        assert app.notes_service.calls[-1]["note_id"] == "n-1"


@pytest.mark.asyncio
async def test_library_shell_note_editor_keywords_empty_without_notes_service():
    """Absent ``app.notes_service`` (or a service missing the method), the
    editor still opens normally with an empty Keywords input -- the
    enrichment is quiet-best-effort, never a hard requirement to open a
    note (matches the current/pre-enrichment behavior).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        assert screen.query_one("#library-note-keywords", Input).value == ""


@pytest.mark.asyncio
async def test_library_shell_note_back_returns_to_list():
    """The editor's Back action returns to the notes list and clears the
    selected note/detail state."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        screen.query_one("#library-note-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")

        assert screen._library_notes_view == "list"
        assert screen._selected_note_id == ""
        assert screen._library_note_detail is None
        assert not screen.query("#library-note-title")


@pytest.mark.asyncio
async def test_library_shell_notes_rail_reentry_resets_to_list():
    """Leaving the notes editor for another rail row and returning to
    Browse > Notes must land on the list, not the previously-open editor."""
    app = _build_test_app()
    _seed_conversations(
        app, _two_conversations(), notes=_two_notes(), media=_two_media_items()
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-row-0")

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")

        assert not screen.query("#library-note-title")
        assert screen._library_notes_view == "list"
        assert screen._selected_note_id == ""
        assert screen._library_note_detail is None


@pytest.mark.asyncio
async def test_library_shell_note_detail_race_discards_stale_fetch():
    """A detail fetch that completes for a no-longer-selected note id must be
    discarded instead of overwriting the currently-open editor."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        for _ in range(150):
            detail = screen._library_note_detail
            if isinstance(detail, dict) and str(detail.get("id")) == "n-1":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("n-1 detail never loaded.")
        assert screen._selected_note_id == "n-1"

        # Simulate a slower in-flight fetch for the previously-selected n-2
        # completing now: it must not overwrite n-1's detail.
        await screen._refresh_library_note_detail("n-2")

        detail = screen._library_note_detail
        assert isinstance(detail, dict)
        assert str(detail.get("id")) == "n-1"


class _DelayedSaveLibraryNotesScopeService(StaticLibraryNotesScopeService):
    """A ``save_note`` that stalls before delegating, so a test can observe
    "the seam call started" as an event distinct from "the save resolved" --
    proving the caller actually awaited the save before doing anything
    else, rather than merely calling it eventually.

    Two stall mechanisms are supported: a fixed ``delay`` (the default --
    good enough when a test only needs "this save takes a moment"), or an
    explicit ``release_event`` (a ``threading.Event``) when a test needs to
    hold a save open indefinitely and release multiple concurrent calls
    together on demand -- e.g. proving an autosave and a flush's save don't
    race each other. The blocking ``.wait()`` is dispatched via
    ``asyncio.to_thread``, mirroring ``_GatedSearchLibraryNotesScopeService``:
    ``save_note`` runs with ``isolate_in_worker=True``, i.e. inside its own
    thread with its own event loop, so an ``asyncio.Event`` set from the
    test's thread/loop would not safely wake a waiter parked on that other
    loop.
    """

    def __init__(self, notes, *, delay: float = 0.15, release_event: threading.Event | None = None):
        super().__init__(notes)
        self.delay = delay
        self.release_event = release_event
        self.save_started: list[str] = []

    async def save_note(self, **kwargs):
        self.save_started.append(str(kwargs.get("note_id") or ""))
        if self.release_event is not None:
            await asyncio.to_thread(self.release_event.wait)
        else:
            await asyncio.sleep(self.delay)
        return await super().save_note(**kwargs)


async def _open_note_editor(screen, pilot, note_id_suffix: str = "n-1"):
    """Drive the shared path to the in-canvas note editor for ``_two_notes()``'s
    first row, then let the mount-time armed-flag callback settle.
    """
    screen.query_one("#library-row-browse-notes").press()
    await _wait_for_selector(screen, pilot, "#library-notes-row-0")
    screen.query_one("#library-notes-row-0").press()
    await _wait_for_selector(screen, pilot, "#library-note-title")
    await pilot.pause()
    await pilot.pause()


def _bump_note_version_externally(service, note_id: str, **field_overrides) -> None:
    """Simulate another writer changing a note: bump its stored version (and
    optionally other fields) on the fake, out from under a screen that still
    has the old version cached -- the setup for every conflict pilot below.
    """
    service.notes = tuple(
        dict(note, version=int(note["version"]) + 1, **field_overrides)
        if note.get("id") == note_id
        else note
        for note in service.notes
    )


def _remove_note_externally(service, note_id: str) -> None:
    """Simulate the note being deleted elsewhere entirely (not just a
    version bump): a subsequent ``get_note_detail`` for it resolves to
    ``None`` on the fake, mirroring a real deleted/never-existed note.
    """
    service.notes = tuple(note for note in service.notes if note.get("id") != note_id)


@pytest.mark.asyncio
async def test_library_shell_note_explicit_save_calls_seam_and_bumps_version():
    """Pressing Save calls the seam with the sanitized fields/keywords list,
    bumps the in-memory version from the seam's dict result, updates the
    meta line to show "saved", and never recomposes the editor -- the
    ``TextArea`` instance must be the exact same object before and after.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        body_before_save = screen.query_one("#library-note-body", TextArea)
        body_before_save.text = "alpha budget line, edited"
        screen.query_one("#library-note-keywords", Input).value = "alpha, beta"
        await pilot.pause()

        screen.query_one("#library-note-save").press()

        service = app.notes_scope_service
        for _ in range(150):
            if service.save_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Save never called the save_note seam.")

        call = service.save_calls[-1]
        assert call["scope"] == "local_note"
        assert call["note_id"] == "n-1"
        assert call["version"] == 2
        assert call["title"] == "Q3 retro"
        assert call["content"] == "alpha budget line, edited"
        assert call["keywords"] == ["alpha", "beta"]

        for _ in range(150):
            if screen._library_note_version == 3:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(f"Version never bumped (still {screen._library_note_version!r}).")

        assert screen.query_one("#library-note-body", TextArea) is body_before_save
        meta = str(screen.query_one("#library-note-meta").renderable)
        assert "saved" in meta


@pytest.mark.asyncio
async def test_library_shell_note_save_patches_detail_title_and_content_for_later_recompose():
    """A save only ever updates the ``#library-note-meta`` ``Static`` in
    place -- it never recomposes the editor itself. But a *later*
    recompose (entering the delete-confirm state, in this test) rebuilds
    the editor's ``TextArea``/``Input`` values fresh from
    ``_library_note_detail``. If Save only patched the cached ``version``
    there and left ``title``/``content`` stale, that later recompose would
    silently revert the just-saved edit back to the pre-save text on
    screen.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "edited body text"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "saved":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Save never completed.")

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm-copy")
        screen.query_one("#library-note-delete-cancel").press()
        await pilot.pause()

        assert screen.query_one("#library-note-body", TextArea).text == "edited body text", (
            "A later recompose reverted the saved edit -- the detail mirror "
            "wasn't patched with the saved title/content."
        )


@pytest.mark.asyncio
async def test_library_shell_note_autosave_fires_after_debounce(monkeypatch):
    """Editing the body (no Save press) arms the autosave debounce; once it
    fires, ``save_note`` was called and the meta line shows "saved".
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.05)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, autosaved"
        await pilot.pause()

        service = app.notes_scope_service
        for _ in range(150):
            if service.save_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Autosave never called save_note without a Save press.")

        assert service.save_calls[-1]["content"] == "alpha budget line, autosaved"

        meta = ""
        for _ in range(150):
            meta = str(screen.query_one("#library-note-meta").renderable)
            if "saved" in meta:
                break
            await pilot.pause(0.02)
        assert "saved" in meta


@pytest.mark.asyncio
async def test_library_shell_note_flush_on_back_saves_before_view_switches():
    """Editing the body then immediately pressing Back must flush (await)
    the pending save before the canvas leaves the editor -- proven by
    observing the view is still "editor" while the (deliberately slow)
    save is in flight, and only becomes "list" once it resolves.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, flushed"
        await pilot.pause()

        screen.query_one("#library-note-back").press()
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Back never triggered the flush save.")

        # The save is still sleeping: the view must not have switched yet.
        assert screen._library_notes_view == "editor"

        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Back never completed once the flush resolved.")

        assert service.save_calls, "The flushed save never actually completed."
        assert service.save_calls[-1]["content"] == "alpha budget line, flushed"


@pytest.mark.asyncio
async def test_library_shell_note_flush_on_rail_switch_saves_before_switching():
    """Editing the body then immediately switching rail rows must flush the
    pending save before the rail switch takes effect, mirroring the
    flush-on-Back contract for the ``_select_library_rail_row`` exit path.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService(_two_media_items())
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, rail-flushed"
        await pilot.pause()

        screen.query_one("#library-row-browse-media").press()
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Rail switch never triggered the flush save.")

        # The save is still sleeping: the rail switch must not have applied yet.
        assert screen._active_mode == "notes"

        for _ in range(150):
            if screen._active_mode == "media":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Rail switch never completed once the flush resolved.")

        assert service.save_calls, "The flushed save never actually completed."
        assert service.save_calls[-1]["content"] == "alpha budget line, rail-flushed"


@pytest.mark.asyncio
async def test_library_shell_flush_waits_for_inflight_autosave(monkeypatch):
    """Back's exit-flush must wait for an in-flight autosave save instead of
    racing its own inline save against it with the same stale version.

    Without the wait, the debounced autosave's ``save_note`` call and the
    flush's inline ``save_note`` call both carry the same
    (not-yet-bumped) version -- an optimistic-lock version conflict that
    the local backend would only ever raise for *another* writer now fires
    against nobody but the note's own autosave, popping a spurious
    "changed elsewhere" conflict banner and aborting the Back navigation.

    Uses ``_DelayedSaveLibraryNotesScopeService``'s ``release_event`` gate
    so the autosave's ``save_note`` call can be held open indefinitely:
    Back is pressed while it is still in flight, proving the flush does
    not issue a second concurrent ``save_note`` call while waiting, then
    the event is released and the flush is proven to complete cleanly.
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.05)
    app = _build_test_app()
    release_event = threading.Event()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes(), release_event=release_event)
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = "alpha budget line, autosave-inflight"
        await pilot.pause()

        # Let the autosave debounce fire; its save_note blocks on
        # release_event (in its own worker thread) so it is genuinely
        # in-flight when Back is pressed.
        for _ in range(150):
            if service.save_started:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Autosave never called save_note.")
        assert len(service.save_started) == 1

        # The Back handler's flush awaits the in-flight save worker, which
        # blocks the message pump -- so ``pilot.pause`` cannot observe the
        # intermediate "flush is waiting" state (it waits for an idle pump
        # that only arrives once the wait completes). Release the gate from
        # a separate thread, independent of the loop, so the in-flight save
        # can finish and the whole Back->flush->navigate sequence settles;
        # then assert on the OUTCOME.
        threading.Timer(0.2, release_event.set).start()
        screen.query_one("#library-note-back").press()

        for _ in range(300):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Back never completed after the in-flight autosave resolved "
                f"(view={screen._library_notes_view!r}, "
                f"autosave_state={screen._library_note_autosave_state!r}, "
                f"save_started={service.save_started!r})."
            )

        # No spurious self-conflict: nobody else touched the note, so the
        # flush must not have raced its own save against the autosave with a
        # stale version.
        assert not screen.query("#library-note-conflict-overwrite"), (
            "A spurious self-conflict was raised even though nobody else "
            "touched the note."
        )
        assert screen._library_note_autosave_state != "conflict"
        # The autosave's save completed; the flush then saw dirty cleared and
        # skipped its own redundant save -> exactly one save_note call, at the
        # note's actual stored version (2), never a stale-version second call.
        assert service.save_calls, "The in-flight autosave's save never completed."
        assert len(service.save_started) == 1, (
            "The flush raced a second concurrent save_note against the "
            f"in-flight autosave (save_started={service.save_started!r})."
        )
        assert service.save_calls[0]["version"] == 2


@pytest.mark.asyncio
async def test_library_shell_note_save_result_after_switch_is_discarded():
    """A save started for note A that only resolves after the user has
    already switched to note B's editor must not land its version bump or
    "saved" meta status on B -- the same stale-result discipline
    ``_refresh_library_note_detail`` and ``_resolve_library_note_conflict``
    already apply to their own out-of-order results.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)  # opens n-1 (seeded at version 2)

        assert screen._selected_note_id == "n-1"
        assert screen._library_note_version == 2

        # Arm a save for n-1 whose fake response is deliberately delayed, but
        # never dirty the editor -- the save is triggered purely by the Save
        # button so ``_library_note_dirty`` stays False and a subsequent row
        # switch's flush is a no-op, letting the switch complete immediately
        # while this save is still in flight.
        screen.query_one("#library-note-save").press()
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError("Save never called the save_note seam.")

        # Switch to n-2's editor while n-1's save is still sleeping. The
        # notes canvas only renders row buttons in its list view, so this
        # goes via Back (a no-op flush, since nothing is dirty) then a row
        # press -- the same "row press" switch path a user takes.
        screen.query_one("#library-note-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-1")
        screen.query_one("#library-notes-row-1").press()
        for _ in range(150):
            if screen._selected_note_id == "n-2":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Row switch to n-2 never completed.")

        for _ in range(150):
            detail = screen._library_note_detail
            if isinstance(detail, dict) and str(detail.get("id")) == "n-2":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("n-2 detail never loaded.")

        assert screen._library_note_version == 1  # n-2's own seeded version
        n2_meta_before = str(screen.query_one("#library-note-meta").renderable)
        assert "saved" not in n2_meta_before

        # Now let the stale n-1 save resolve and give the (buggy) mutation a
        # few pause cycles to land, if it were going to.
        for _ in range(150):
            if service.save_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The stale n-1 save never resolved.")
        for _ in range(10):
            await pilot.pause(0.02)

        assert screen._selected_note_id == "n-2"
        assert screen._library_note_version == 1, (
            "n-1's stale save result clobbered n-2's version: "
            f"{screen._library_note_version!r}"
        )
        n2_meta_after = str(screen.query_one("#library-note-meta").renderable)
        assert "saved" not in n2_meta_after, (
            f"n-1's stale save result leaked into n-2's meta line: {n2_meta_after!r}"
        )


@pytest.mark.asyncio
async def test_library_shell_note_conflict_shows_overwrite_reload_and_keeps_user_text():
    """A version conflict (the seam returns ``False``) stops the autosave
    timer, shows the Overwrite/Reload actions, and re-seeds the editor from
    the user's own kept text -- never the stale server-side detail.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-body", TextArea).text = "kept text that must survive"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        assert screen.query("#library-note-conflict-overwrite")
        assert screen.query("#library-note-conflict-reload")
        meta = str(screen.query_one("#library-note-meta").renderable)
        assert "changed elsewhere" in meta
        assert screen.query_one("#library-note-body", TextArea).text == (
            "kept text that must survive"
        )


@pytest.mark.asyncio
async def test_library_shell_note_conflict_during_preview_reads_live_text():
    """A save conflict raised while Preview is toggled on must not leave a
    stale ``_library_note_preview``/``_library_note_preview_snapshot`` behind:
    the conflict UI always renders the live ``TextArea`` (``compose`` never
    threads ``preview`` through for the conflict branch), so
    ``_read_library_note_editor_fields`` still reading through the old
    preview snapshot would send stale, pre-preview text to the seam on a
    save from the conflict UI -- and visibly revert the user's on-screen
    edits on the next recompose.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service

        screen.query_one("#library-note-body", TextArea).text = "T1"
        await pilot.pause()

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")
        assert screen._library_note_preview is True

        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        # The conflict UI always shows the live TextArea, never the
        # read-only Markdown preview, regardless of the Preview flag.
        conflict_body = screen.query_one("#library-note-body", TextArea)
        assert conflict_body.text == "T1"

        conflict_body.text = "T2"
        await pilot.pause()

        calls_before_second_save = len(service.save_calls)
        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if len(service.save_calls) > calls_before_second_save:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The second Save press never called the seam.")
        # Give the resulting conflict recompose a few cycles to settle.
        for _ in range(10):
            await pilot.pause(0.02)

        assert service.save_calls[-1]["content"] == "T2", (
            "The conflict-UI save sent stale pre-preview text to the seam: "
            f"{service.save_calls[-1]['content']!r}"
        )
        assert screen.query_one("#library-note-body", TextArea).text == "T2", (
            "The second conflict recompose reverted the user's on-screen edit."
        )


@pytest.mark.asyncio
async def test_library_shell_note_conflict_overwrite_resaves_with_fresh_version():
    """Overwrite re-fetches the note silently, takes only the fresh
    version, and re-saves the user's kept text at that version -- ending
    with the fake holding the user's text at the newest version.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-body", TextArea).text = "kept text that must survive"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        (await _wait_for_selector(screen, pilot, "#library-note-conflict-overwrite")).press()
        for _ in range(150):
            if screen._library_note_autosave_state == "saved":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Overwrite never completed.")

        assert not screen.query("#library-note-conflict-overwrite")
        stored = next(note for note in service.notes if note["id"] == "n-1")
        assert stored["content"] == "kept text that must survive"
        assert stored["version"] == 4  # v2 -> externally bumped to v3 -> overwrite saves to v4
        assert screen._library_note_version == 4


@pytest.mark.asyncio
async def test_library_shell_note_conflict_reload_discards_local_edits():
    """Reload discards the local edit and recomposes the editor from the
    freshly re-fetched server-side detail.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(
            service, "n-1", title="Server-side title", content="Server-side content"
        )

        screen.query_one("#library-note-body", TextArea).text = "local edits to discard"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        (await _wait_for_selector(screen, pilot, "#library-note-conflict-reload")).press()
        for _ in range(150):
            if (
                screen._library_note_autosave_state == "idle"
                and screen._library_notes_view == "editor"
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Reload never completed.")

        assert not screen.query("#library-note-conflict-reload")
        assert screen.query_one("#library-note-body", TextArea).text == "Server-side content"
        assert screen.query_one("#library-note-title", Input).value == "Server-side title"


@pytest.mark.asyncio
async def test_library_shell_note_conflict_reload_falls_back_to_list_when_note_missing():
    """A Reload from the conflict UI whose silent re-fetch discovers the
    note was deleted elsewhere entirely (not just version-bumped again)
    must fall back to the list view with a warning -- not strand the
    canvas on a permanent "Loading note…" placeholder, mirroring the
    missing-note fallback ``_refresh_library_note_detail`` already applies.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-body", TextArea).text = "kept text"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        # The note is now gone entirely (not just bumped again) before
        # Reload's silent re-fetch runs.
        _remove_note_externally(service, "n-1")

        (await _wait_for_selector(screen, pilot, "#library-note-conflict-reload")).press()
        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Reload never fell back to the list view for a missing note "
                f"(stuck: view={screen._library_notes_view!r}, "
                f"autosave_state={screen._library_note_autosave_state!r})."
            )

        assert screen._selected_note_id == ""
        assert screen._library_note_detail is None
        assert screen._library_note_autosave_state == "idle"
        assert not screen.query("#library-note-conflict-reload")


@pytest.mark.asyncio
async def test_library_shell_note_conflict_overwrite_falls_back_to_list_when_note_missing():
    """Same fallback as the Reload case above, but for Overwrite: it also
    runs the same silent re-fetch first, so a note deleted elsewhere must
    not leave Overwrite stuck silently no-op-ing against a conflict UI that
    can never be resolved.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        _bump_note_version_externally(service, "n-1")  # now stored at v3; screen still has v2

        screen.query_one("#library-note-body", TextArea).text = "kept text"
        await pilot.pause()

        screen.query_one("#library-note-save").press()
        for _ in range(150):
            if screen._library_note_autosave_state == "conflict":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The version conflict was never reached.")

        _remove_note_externally(service, "n-1")

        (await _wait_for_selector(screen, pilot, "#library-note-conflict-overwrite")).press()
        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Overwrite never fell back to the list view for a missing note "
                f"(stuck: view={screen._library_notes_view!r}, "
                f"autosave_state={screen._library_note_autosave_state!r})."
            )

        assert screen._selected_note_id == ""
        assert screen._library_note_detail is None
        assert screen._library_note_autosave_state == "idle"
        assert not screen.query("#library-note-conflict-overwrite")


@pytest.mark.asyncio
async def test_library_shell_note_delete_shows_inline_confirm_without_deleting():
    """Pressing Delete shows the inline confirm affordance, not an immediate delete."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm-copy")

        assert screen._library_note_confirming_delete is True
        assert screen.query_one("#library-note-delete-confirm")
        assert screen.query_one("#library-note-delete-cancel")
        assert not screen.query("#library-note-save")
        assert not screen.query("#library-note-delete")

        service = app.notes_scope_service
        assert service.delete_calls == []
        assert screen._library_notes_view == "editor"


@pytest.mark.asyncio
async def test_library_shell_note_delete_confirm_removes_note_and_returns_to_list():
    """Confirming the delete calls the seam with the current version, drops
    the note from the list view, and resets the editor state."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")

        screen.query_one("#library-note-delete-confirm").press()

        service = app.notes_scope_service
        for _ in range(150):
            if service.delete_calls and screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Delete never completed. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert service.delete_calls, "delete_note was never called"
        assert service.delete_calls[-1]["scope"] == "local_note"
        assert service.delete_calls[-1]["note_id"] == "n-1"
        assert service.delete_calls[-1]["version"] == 2
        assert screen._library_note_confirming_delete is False
        assert screen._selected_note_id == ""
        assert screen._library_notes_view == "list"
        assert not screen.query("#library-note-title")
        assert not any(
            "Q3 retro" in str(getattr(button, "label", ""))
            for button in screen.query(".library-notes-row")
        )


@pytest.mark.asyncio
async def test_library_shell_note_delete_cancel_leaves_note_intact():
    """Cancelling the inline confirm discards it without calling the service."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")

        screen.query_one("#library-note-delete-cancel").press()
        await pilot.pause()
        await pilot.pause()

        assert screen._library_note_confirming_delete is False
        assert not screen.query("#library-note-delete-confirm")
        assert not screen.query("#library-note-delete-confirm-copy")
        assert screen.query_one("#library-note-delete")
        assert screen.query_one("#library-note-title", Input).value == "Q3 retro"

        service = app.notes_scope_service
        assert service.delete_calls == []


@pytest.mark.asyncio
async def test_library_shell_note_delete_confirm_does_not_arm_autosave(monkeypatch):
    """Entering (and leaving) the delete-confirmation state recomposes the
    editor, which remounts the Input/TextArea and re-fires their mount-time
    ``Changed`` events. Those must never be mistaken for a real edit and
    fire a background autosave while a destructive-action confirmation is
    on screen -- the same armed-flag discipline every other note-editor
    recompose site already follows (see ``_refresh_library_note_detail``,
    ``_save_library_note``'s conflict branch, and
    ``_resolve_library_note_conflict``).
    """
    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTES_AUTOSAVE_SECONDS", 0.05)
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        calls_before_delete = len(service.save_calls)

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm-copy")

        # Wait well past the (monkeypatched) debounce window -- long enough
        # for a spurious autosave timer to have fired if the armed flag
        # were left set across the recompose.
        for _ in range(150):
            await pilot.pause(0.02)

        assert len(service.save_calls) == calls_before_delete, (
            "Entering delete-confirm must not trigger an autosave."
        )
        assert screen._library_note_confirming_delete is True
        assert screen.query_one("#library-note-delete-confirm")
        assert screen.query_one("#library-note-delete-cancel")

        screen.query_one("#library-note-delete-cancel").press()
        await pilot.pause()

        for _ in range(150):
            await pilot.pause(0.02)

        assert len(service.save_calls) == calls_before_delete, (
            "Cancelling delete-confirm must not trigger an autosave either."
        )
        assert screen._library_note_confirming_delete is False
        assert not screen.query("#library-note-delete-confirm")
        assert screen.query_one("#library-note-delete")


@pytest.mark.asyncio
async def test_library_shell_note_delete_stale_version_rearms_the_editor():
    """Task 7 regression rider: the delete confirm's stale-version failure
    path must re-arm dirty-tracking around its recompose, the same
    ``_library_note_editor_armed = False`` +
    ``call_after_refresh(_arm_library_note_editor)`` dance every other
    notes-editor recompose already uses (before this rider fix, this one
    call site skipped it). Proven deterministically via a spy on
    ``_arm_library_note_editor`` rather than by observing autosave timing:
    a worker-triggered recompose racing its own mount-time
    ``Input``/``TextArea`` ``Changed`` events against the re-arm is a
    pre-existing characteristic of this mechanism (independently
    reproducible today on the already-shipped save-conflict recompose in
    ``_save_library_note``), so asserting "no autosave ever fires" around
    the recompose itself would be asserting a guarantee this codebase
    doesn't actually make -- not something introduced or fixable within
    this rider's narrow scope (adding the same dance already used
    elsewhere to 3 more call sites).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        service = app.notes_scope_service
        # Another writer deletes/updates n-1 out from under the open editor,
        # bumping its stored version so the confirmed delete below is stale.
        _bump_note_version_externally(service, "n-1")

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")
        await pilot.pause()

        # Spy on the re-arm hook from here on, so only calls triggered by
        # the confirmed (stale) delete below count -- entering confirm mode
        # already re-arms once on its own (a separate, already-covered path).
        rearm_calls: list[bool] = []
        original_arm = screen._arm_library_note_editor

        def _spy_arm() -> None:
            rearm_calls.append(True)
            original_arm()

        screen._arm_library_note_editor = _spy_arm

        screen.query_one("#library-note-delete-confirm").press()

        for _ in range(150):
            if service.delete_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Delete confirm never called the delete_note seam.")

        # The delete lost the optimistic-lock race (falsy result): the note
        # must still be open in the editor, not removed.
        for _ in range(150):
            if not screen._library_note_confirming_delete:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Stale-version delete never returned to the normal action row.")
        await pilot.pause()

        assert screen._library_notes_view == "editor"
        assert screen._selected_note_id == "n-1"
        assert screen.query_one("#library-note-delete")
        assert rearm_calls, (
            "The stale-version delete failure path never re-armed the "
            "editor's dirty-tracking after its recompose."
        )
        assert screen._library_note_editor_armed is True


@pytest.mark.asyncio
async def test_library_shell_filtered_delete_refreshes_list_without_ghost():
    """Deleting a note opened from a filtered list must clear the active
    filter and refresh from the full snapshot, so the deleted note does
    not linger as a ghost row in the (now-stale) filtered results
    (final-review finding, part A).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-filter")
        box = screen.query_one("#library-notes-filter", Input)
        box.value = "retro"
        box.focus()
        await pilot.pause()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")

        # The filter narrows the list to just "Q3 retro" (n-1).
        assert not any(
            "Reading list" in str(getattr(button, "label", ""))
            for button in screen.query(".library-notes-row")
        )
        assert screen._library_notes_filter == "retro"

        screen.query_one("#library-notes-row-0").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        assert screen._selected_note_id == "n-1"

        screen.query_one("#library-note-delete").press()
        await _wait_for_selector(screen, pilot, "#library-note-delete-confirm")
        screen.query_one("#library-note-delete-confirm").press()

        service = app.notes_scope_service
        for _ in range(150):
            if service.delete_calls and screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Delete never completed. Visible text: {_visible_text(screen)}"
            )

        rows_text: list[str] = []
        for _ in range(150):
            rows_text = [
                str(getattr(button, "label", ""))
                for button in screen.query(".library-notes-row")
            ]
            if any("Reading list" in text for text in rows_text) and not any(
                "Q3 retro" in text for text in rows_text
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"List never settled on the refreshed (unfiltered) snapshot: {rows_text}"
            )

        assert screen._library_notes_filter == ""
        assert screen._library_notes_filter_records is None
        filter_box = screen.query_one("#library-notes-filter", Input)
        assert filter_box.value == ""
        assert not any("Q3 retro" in text for text in rows_text), (
            f"Deleted note still rendered as a ghost row: {rows_text}"
        )
        assert any("Reading list" in text for text in rows_text), (
            f"Remaining note missing from the refreshed list: {rows_text}"
        )


@pytest.mark.asyncio
async def test_library_shell_opening_missing_note_falls_back_to_list():
    """Pressing a row whose note has since vanished (a ghost row, or a note
    deleted elsewhere) must fall back to the list view instead of stranding
    the canvas on the "Loading note…" placeholder forever (final-review
    finding, part B).
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-row-0")
        row_button = screen.query_one("#library-notes-row-0")
        assert row_button.note_id == "n-1"

        # Simulate the note vanishing out from under the still-rendered row
        # (deleted elsewhere, or simply a stale/ghost row): the fake's
        # ``get_note_detail`` now resolves to ``None`` for "n-1".
        service = app.notes_scope_service
        service.notes = tuple(
            note for note in service.notes if note.get("id") != "n-1"
        )

        row_button.press()

        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"Never fell back to the list. Visible text: {_visible_text(screen)}"
            )
        await pilot.pause()

        assert not screen.query("#library-note-loading"), (
            "Stuck on the permanent 'Loading note…' placeholder."
        )
        assert not screen.query("#library-note-title")
        assert screen._library_note_detail is None
        assert screen._selected_note_id == ""
        app.notify.assert_called_once()
        assert app.notify.call_args.kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_note_preview_toggle_shows_markdown_and_restores_edit():
    """Preview swaps the body TextArea for a read-only Markdown widget
    rendering the *current* (unsaved) text; toggling back to Edit restores
    the TextArea with that same edit intact.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        body = screen.query_one("#library-note-body", TextArea)
        body.text = "alpha budget line, unsaved preview edit"
        await pilot.pause()

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")

        assert not screen.query("#library-note-body")
        preview = screen.query_one("#library-note-preview-body", Markdown)
        assert "alpha budget line, unsaved preview edit" in preview.source
        assert str(screen.query_one("#library-note-preview").label) == "Edit"
        # Title/keywords stay live Inputs -- only the body swaps.
        assert screen.query_one("#library-note-title", Input).value == "Q3 retro"

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-body")

        assert not screen.query("#library-note-preview-body")
        restored_body = screen.query_one("#library-note-body", TextArea)
        assert restored_body.text == "alpha budget line, unsaved preview edit"
        assert str(screen.query_one("#library-note-preview").label) == "Preview"


@pytest.mark.asyncio
async def test_library_shell_note_save_works_while_previewing():
    """Pressing Save while Preview is on must still persist the edit.

    While Preview is showing, ``#library-note-body`` is a read-only
    ``Markdown`` widget -- not the ``TextArea`` -- so a naive
    ``query_one("#library-note-body", TextArea)`` raises ``NoMatches``.
    ``_save_library_note`` must instead read through the preview-aware
    ``_read_library_note_editor_fields`` helper (the same one Export/Copy/
    Use-in-Console already rely on) so Save is not a silent no-op while
    previewing.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = (
            "alpha budget line, edited while about to preview"
        )
        await pilot.pause()

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")
        assert not screen.query("#library-note-body")

        screen.query_one("#library-note-save").press()

        service = app.notes_scope_service
        for _ in range(150):
            if service.save_calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Save never called the save_note seam while previewing."
            )

        call = service.save_calls[-1]
        assert call["content"] == "alpha budget line, edited while about to preview"

        for _ in range(150):
            if not screen._library_note_dirty:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Save while previewing never cleared the dirty flag.")


@pytest.mark.asyncio
async def test_library_shell_note_back_flushes_while_previewing():
    """Back must flush a pending edit even while Preview is still on.

    Regression pilot for the "silent data loss" finding: edit the body,
    toggle Preview on (without toggling back to Edit), then press Back
    immediately -- the edit must reach the seam before the view switches
    to the notes list, mirroring the flush-on-Back contract already
    proven for the non-preview case.
    """
    app = _build_test_app()
    service = _DelayedSaveLibraryNotesScopeService(_two_notes())
    app.notes_scope_service = service
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(_two_conversations())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-body", TextArea).text = (
            "alpha budget line, edited then previewed then back"
        )
        await pilot.pause()

        screen.query_one("#library-note-preview").press()
        await _wait_for_selector(screen, pilot, "#library-note-preview-body")
        assert not screen.query("#library-note-body")

        screen.query_one("#library-note-back").press()
        for _ in range(50):
            if service.save_started:
                break
            await pilot.pause(0.01)
        else:
            raise AssertionError(
                "Back never triggered the flush save while previewing."
            )

        # The save is still sleeping: the view must not have switched yet.
        assert screen._library_notes_view == "editor"

        for _ in range(150):
            if screen._library_notes_view == "list":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                "Back never completed once the previewing flush resolved."
            )

        assert service.save_calls, (
            "The flushed save while previewing never actually completed."
        )
        assert service.save_calls[-1]["content"] == (
            "alpha budget line, edited then previewed then back"
        )


@pytest.mark.asyncio
async def test_library_shell_note_export_markdown_pushes_file_save_dialog():
    """Export .md pushes a ``FileSave`` dialog pre-filled with a sanitized
    default filename derived from the note's current title."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-export-md").press()
        for _ in range(150):
            if isinstance(host.screen_stack[-1], FileSave):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Export .md never pushed a FileSave dialog.")

        dialog = host.screen_stack[-1]
        assert dialog._default_file == "Q3 retro.md"

        await host.pop_screen()
        await pilot.pause()


@pytest.mark.asyncio
async def test_library_shell_note_write_export_file_writes_expected_content(tmp_path):
    """The export write-path (bypassing the dialog UI, which is exercised
    separately above) writes the same content ``build_note_export_content``
    produces and notifies on success -- the "pure part" the brief calls out
    as pilot-testable independent of driving the real file dialog.

    ``_write_library_note_export_file`` is a plain (non-async) method --
    it awaits nothing -- so this calls it directly rather than awaiting it.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        destination = tmp_path / "export.md"
        screen._write_library_note_export_file(
            destination, "markdown", "Q3 retro", "alpha budget line", "alpha, beta", "n-1"
        )

        written = destination.read_text(encoding="utf-8")
        assert "title: Q3 retro" in written
        assert "keywords: alpha, beta" in written
        assert "note_id: n-1" in written
        assert written.endswith("alpha budget line")
        app.notify.assert_called_once()
        assert "exported successfully" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_library_shell_note_write_export_file_rejects_invalid_path(tmp_path, monkeypatch):
    """A ``FileSave``-returned path that fails ``validate_path_simple``
    must be rejected with a quiet warning notice -- no write, no crash --
    rather than trusting the dialog's returned path unconditionally.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    def _reject_path(*_args, **_kwargs):
        raise ValueError("rejected for test")

    monkeypatch.setattr(library_screen_module, "validate_path_simple", _reject_path)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        destination = tmp_path / "export.md"
        screen._write_library_note_export_file(
            destination, "markdown", "Q3 retro", "alpha budget line", "alpha, beta", "n-1"
        )

        assert not destination.exists()
        app.notify.assert_called_once()
        args, kwargs = app.notify.call_args
        assert "Rejected export path" in args[0]
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_note_copy_calls_clipboard_seam():
    """Copy calls the app's ``copy_to_clipboard`` seam with the markdown
    export shape and notifies on success."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.copy_to_clipboard = Mock()
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-copy").press()
        await pilot.pause()

    app.copy_to_clipboard.assert_called_once()
    copied = app.copy_to_clipboard.call_args.args[0]
    assert copied.startswith("---\n")
    assert "title: Q3 retro" in copied
    assert "alpha budget line" in copied
    app.notify.assert_called_once()
    assert "copied to clipboard" in app.notify.call_args.args[0]


@pytest.mark.asyncio
async def test_library_shell_note_use_in_console_triggers_handoff():
    """"Use in Console" stages the open note as Console context, mirroring
    the media viewer's "Use in Chat" handoff test.
    """
    app = _build_test_app()
    note_items = _two_notes()[:1]
    _seed_conversations(app, [], notes=note_items)
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("note", "n-1", "Q3 retro"),),
    )
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_note_editor(screen, pilot)

        screen.query_one("#library-note-use-in-console").press()
        await pilot.pause()
        await pilot.pause()

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "notes"
    assert payload.item_type == "note"
    assert payload.source_id == "n-1"
    assert payload.title == "Q3 retro"
    assert "alpha budget line" in payload.body
    assert payload.metadata["note_version"] == 2


@pytest.mark.asyncio
async def test_library_shell_note_use_in_console_without_open_note_notifies():
    """No handoff fires when no note is currently open in the editor."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=_two_notes())
    app.open_chat_with_handoff = Mock()
    app.notify = Mock()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        assert screen._selected_note_id == ""
        screen._open_selected_library_note_handoff()
        await pilot.pause()

    app.open_chat_with_handoff.assert_not_called()
    app.notify.assert_called_once()
    message = app.notify.call_args.args[0]
    assert "Open a note" in message


def test_library_note_css_bounds_editor_body_and_mutes_meta():
    """Generated-CSS presence check (house pattern: see
    ``test_library_source_actions_use_console_text_control_style``) --
    the note editor's ``TextArea`` must never be left at Textual's default
    ``height: 1fr`` inside the scrolling Library canvas, and the meta line
    must use the same muted tone as the media viewer's.
    """
    source_css = Path("tldw_chatbook/css/components/_agentic_terminal.tcss").read_text(encoding="utf-8")
    bundled_css = Path("tldw_chatbook/css/tldw_cli_modular.tcss").read_text(encoding="utf-8")
    for css in (source_css, bundled_css):
        assert "#library-note-body {" in css
        body_block = css[css.index("#library-note-body {"):]
        body_block = body_block[: body_block.index("}")]
        assert "height: auto;" in body_block
        assert "min-height: 12;" in body_block
        assert "max-height: 20;" in body_block

        assert "#library-note-meta {" in css
        meta_block = css[css.index("#library-note-meta {"):]
        meta_block = meta_block[: meta_block.index("}")]
        assert "color: $ds-text-muted;" in meta_block


@pytest.mark.asyncio
async def test_library_shell_create_note_row_renders_blank_and_template_rows():
    """Selecting the rail's Create > New note row renders the Blank note
    action plus at least one template row, carrying a ``template_key``."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        assert screen.query_one("#library-notes-create-blank")
        template_row = screen.query_one("#library-notes-template-0")
        assert getattr(template_row, "template_key", None)


@pytest.mark.asyncio
async def test_library_shell_create_blank_note_lands_in_editor():
    """Pressing Blank note calls ``save_note`` with ``note_id=None`` and
    ``title="Untitled"``, then opens the in-canvas editor on the newly
    created note and refreshes the notes rail count."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        screen.query_one("#library-notes-create-blank").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        service = app.notes_scope_service
        assert service.save_calls, "Blank note create never called the seam."
        call = service.save_calls[-1]
        assert call["note_id"] is None
        assert call["title"] == "Untitled"
        assert call["content"] == ""
        assert call["keywords"] is None

        assert screen._library_notes_view == "editor"
        assert screen._library_selected_row_id == "browse-notes"
        created_note = next(n for n in service.notes if n["title"] == "Untitled")
        assert screen._selected_note_id == created_note["id"]
        assert screen.query_one("#library-note-title", Input).value == "Untitled"

        for _ in range(150):
            if screen._local_source_counts.get("notes") == 3:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The notes snapshot/rail count never refreshed after create.")
        rail_label = str(screen.query_one("#library-row-browse-notes").label)
        assert "(3)" in rail_label


@pytest.mark.asyncio
async def test_library_shell_create_from_template_uses_template_fields():
    """Pressing a template row creates a note using that template's title/
    content with ``{date}``/``{time}``/``{datetime}`` placeholders resolved
    to the current date/time -- mirroring the standalone Notes screen's
    ``_create_local_note_from_template`` substitution semantics."""
    from datetime import datetime

    from tldw_chatbook.Event_Handlers.notes_events import NOTE_TEMPLATES

    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        template_button = next(
            button
            for button in screen.query(".library-notes-template-row")
            if getattr(button, "template_key", None) == "meeting"
        )
        expected = NOTE_TEMPLATES["meeting"]
        # Capture the date both before and after pressing the row so a
        # midnight rollover mid-test can't make this flaky.
        before = datetime.now().strftime("%Y-%m-%d")
        template_button.press()
        await _wait_for_selector(screen, pilot, "#library-note-title")
        after = datetime.now().strftime("%Y-%m-%d")

        service = app.notes_scope_service
        assert service.save_calls, "Template create never called the seam."
        call = service.save_calls[-1]
        assert call["note_id"] is None
        assert "{date}" not in call["title"]
        assert "{date}" not in call["content"]
        assert "{time}" not in call["content"]
        assert call["title"] in (
            expected["title"].format(date=before),
            expected["title"].format(date=after),
        )
        assert re.search(r"\*\*Date:\*\* \d{4}-\d{2}-\d{2}", call["content"])
        assert re.search(r"\*\*Time:\*\* \d{2}:\d{2}", call["content"])
        # Template keywords ride along on create -- the standalone screen
        # applies them, so Library parity requires them at the seam too.
        assert call["keywords"] == ["meeting", "notes"]
        assert screen._library_notes_view == "editor"


def _fake_import_dialog_result(screen, selected_path):
    """Monkeypatch ``screen.app.push_screen`` so the Import note dialog's
    callback fires immediately with ``selected_path`` instead of driving
    the real ``FileOpen`` file-browser UI (which needs keystrokes into a
    path input and is unrelated to what this handler contract tests: that
    a *resolved* dialog result reaches ``_import_library_note_from_path``).
    """
    calls = []

    def _fake_push_screen(dialog, callback=None):
        calls.append(dialog)
        if callback is not None:
            screen.run_worker(callback(selected_path))
        return None

    screen.app.push_screen = _fake_push_screen
    return calls


@pytest.mark.asyncio
async def test_library_shell_import_note_lands_in_editor(tmp_path):
    """Pressing Import note, with the dialog resolving to a real file, reads
    that file's title/content and creates a note through the same
    ``_create_library_note`` seam the Blank note/template rows use --
    landing in the editor with the snapshot/count refresh that seam already
    performs."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    note_file = tmp_path / "imported.json"
    note_file.write_text(
        json.dumps({"title": "Imported from disk", "content": "body from disk"}),
        encoding="utf-8",
    )

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        push_calls = _fake_import_dialog_result(screen, note_file)

        screen.query_one("#library-notes-import").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        assert push_calls and isinstance(push_calls[0], FileOpen)

        service = app.notes_scope_service
        assert service.save_calls, "Import never called the create seam."
        call = service.save_calls[-1]
        assert call["note_id"] is None
        assert call["title"] == "Imported from disk"
        assert call["content"] == "body from disk"

        assert screen._library_notes_view == "editor"
        created_note = next(n for n in service.notes if n["title"] == "Imported from disk")
        assert screen._selected_note_id == created_note["id"]

        for _ in range(150):
            if screen._local_source_counts.get("notes") == 3:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("The notes snapshot/rail count never refreshed after import.")


@pytest.mark.asyncio
async def test_library_shell_import_note_oversize_file_rejected(tmp_path, monkeypatch):
    """A file larger than ``LIBRARY_NOTE_CONTENT_MAX_CHARS`` is rejected with
    a quiet warning notice -- no note is created."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    app.notify = Mock()
    host = LibraryHarness(app)

    monkeypatch.setattr(library_screen_module, "LIBRARY_NOTE_CONTENT_MAX_CHARS", 10)
    note_file = tmp_path / "too_big.txt"
    note_file.write_text("x" * 50, encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        _fake_import_dialog_result(screen, note_file)

        service = app.notes_scope_service
        screen.query_one("#library-notes-import").press()
        for _ in range(60):
            await pilot.pause(0.02)

        assert not service.save_calls, "Oversize import must not create a note."
        assert screen._library_notes_view != "editor"
        app.notify.assert_called()
        args, kwargs = app.notify.call_args
        assert "import" in args[0].lower()
        assert kwargs.get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_import_note_md_without_title_uses_filename_stem(tmp_path):
    """A ``.md`` file (which never carries a JSON/YAML "title" key in this
    parser's contract) falls back to the filename stem as the note title."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    note_file = tmp_path / "My Great Notes.md"
    note_file.write_text("# Just a heading\n\nSome body text.", encoding="utf-8")

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        _fake_import_dialog_result(screen, note_file)

        screen.query_one("#library-notes-import").press()
        await _wait_for_selector(screen, pilot, "#library-note-title")

        service = app.notes_scope_service
        call = service.save_calls[-1]
        assert call["title"] == "My Great Notes"
        assert call["content"] == "# Just a heading\n\nSome body text."


@pytest.mark.asyncio
async def test_library_shell_import_note_cancelled_dialog_is_noop():
    """Cancelling the ``FileOpen`` dialog (callback fired with ``None``)
    creates no note and does not crash."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-import")

        _fake_import_dialog_result(screen, None)

        service = app.notes_scope_service
        screen.query_one("#library-notes-import").press()
        for _ in range(30):
            await pilot.pause(0.02)

        assert not service.save_calls
        assert screen._library_notes_view != "editor"


@pytest.mark.asyncio
async def test_library_shell_create_view_groups_templates_without_blank_duplicate():
    """The create view separates Blank note from a "From a template" group,
    excludes the redundant blank template row, and shows each template's
    resolved title as a secondary line."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-create-note").press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")

        # Group label between the Blank action and the template rows.
        section = screen.query_one("#library-notes-template-section")
        assert str(section.renderable) == "From a template"

        template_rows = list(screen.query(".library-notes-template-row"))
        assert template_rows, "No template rows rendered."
        # The blank template row is excluded -- Blank note is the one
        # canonical empty path (it also had a different default title).
        assert all(
            getattr(button, "template_key", "") != "blank" for button in template_rows
        )
        assert "Empty note" not in _visible_text(screen)
        # Rows show the resolved title (date substituted) as a secondary line.
        meeting = next(
            button
            for button in template_rows
            if getattr(button, "template_key", None) == "meeting"
        )
        label = str(meeting.label)
        assert "\n" in label
        assert re.search(r"Meeting Notes - \d{4}-\d{2}-\d{2}", label)
        assert "{date}" not in label


def test_library_note_template_fields_malformed_placeholder_degrades_to_raw_text():
    """A template with an unknown ``{placeholder}`` or an unbalanced brace
    must not raise -- the create flow falls back to the raw (unsubstituted)
    text instead of crashing, mirroring the guarded ``.format()`` call the
    standalone Notes screen uses but degrading gracefully instead of
    aborting the create."""
    template = {
        "title": "Notes - {unknown_key}",
        "content": "Stray brace ahead: { oops",
    }

    title, content = LibraryScreen._library_note_template_fields(template)

    assert title == "Notes - {unknown_key}"
    assert content == "Stray brace ahead: { oops"


# ----- Notes sync panel --------------------------------------------------

class _RecordingSyncResults:
    def __init__(self):
        self.created_notes = []
        self.updated_notes = []
        self.created_files = []
        self.updated_files = []
        self.conflicts = []
        self.errors = []


class _RecordingNotesSyncService:
    """Records the exact args ``sync_folder`` was called with, and lets the
    test control the returned (session_id, results) tuple. The progress
    callback is fired from a real background thread (mirroring the real
    engine's worker-thread callback) and gated on a ``threading.Event`` so
    the test can observe the mid-run targeted status update before letting
    the run complete -- proving the update lands without a recompose.
    """

    instances = []

    def __init__(self, notes_service, db):
        self.notes_service = notes_service
        self.db = db
        self.calls = []
        self.progress_fired = threading.Event()
        self.release_event = threading.Event()
        _RecordingNotesSyncService.instances.append(self)

    async def sync_folder(self, *, root_folder, user_id, direction,
                           conflict_resolution, progress_callback=None, extensions=None):
        self.calls.append({
            "root_folder": root_folder,
            "user_id": user_id,
            "direction": direction,
            "conflict_resolution": conflict_resolution,
        })
        if progress_callback is not None:
            from tldw_chatbook.Notes.sync_engine import SyncProgress
            progress_callback(SyncProgress(total_files=2, processed_files=1))
            self.progress_fired.set()
        # ``_run_library_service_call(..., isolate_in_worker=True)`` already
        # runs this coroutine on a worker thread with no event loop of its
        # own (see library_screen.py); blocking here on a plain
        # threading.Event (not asyncio.sleep) is what actually holds up
        # that worker thread until the test lets it proceed.
        await asyncio.to_thread(self.release_event.wait)
        results = _RecordingSyncResults()
        results.created_notes = ["n-new"]
        return ("session-1", results)


def _prepare_library_notes_sync_app(app, *, notes=None):
    _seed_conversations(app, _two_conversations(), notes=notes or _two_notes())
    app.notes_service = Mock()
    app.chachanotes_db = Mock()
    # The sync direction/conflict/auto-sync config keys live in the real,
    # session-shared CLI config file (under the isolated test HOME) -- a
    # previous test in this same session may have persisted a non-default
    # value there. Reset to known starting values so every sync test's
    # cycling assertions are deterministic regardless of run order.
    from tldw_chatbook.config import save_setting_to_cli_config

    save_setting_to_cli_config("notes", "sync_direction", "bidirectional")
    save_setting_to_cli_config("notes", "sync_conflict_resolution", "newer_wins")
    save_setting_to_cli_config("notes", "auto_sync", False)


async def _open_library_notes_sync_panel(screen, pilot):
    screen.query_one("#library-row-browse-notes").press()
    await _wait_for_selector(screen, pilot, "#library-notes-sync-open")
    screen.query_one("#library-notes-sync-open").press()
    await _wait_for_selector(screen, pilot, "#library-notes-sync-back")


@pytest.mark.asyncio
async def test_library_shell_notes_sync_button_opens_sync_mode():
    """Pressing Sync on the notes list header opens the sync panel with all
    of its widgets present, and Back returns to the notes list."""
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_view == "sync"
        for selector in (
            "#library-notes-sync-back",
            "#library-notes-sync-header",
            "#library-notes-sync-folder",
            "#library-notes-sync-browse",
            "#library-notes-sync-direction",
            "#library-notes-sync-conflict",
            "#library-notes-sync-auto",
            "#library-notes-sync-run",
            "#library-notes-sync-status",
            "#library-notes-sync-activity",
        ):
            assert screen.query_one(selector), f"{selector} missing from sync panel"

        screen.query_one("#library-notes-sync-back").press()
        await _wait_for_selector(screen, pilot, "#library-notes-sync-open")
        assert screen._library_notes_view == "list"
        assert not screen.query("#library-notes-sync-back")


@pytest.mark.asyncio
async def test_library_shell_notes_sync_direction_cycles_and_persists():
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_sync_direction == "bidirectional"
        button = screen.query_one("#library-notes-sync-direction", Button)
        assert "Bidirectional" in str(button.label)

        button.press()
        await pilot.pause()
        assert screen._library_notes_sync_direction == "disk_to_db"
        from tldw_chatbook.config import get_cli_setting
        assert get_cli_setting("notes", "sync_direction", None) == "disk_to_db"
        button = screen.query_one("#library-notes-sync-direction", Button)
        assert "Disk" in str(button.label)


@pytest.mark.asyncio
async def test_library_shell_notes_sync_conflict_cycles_and_persists():
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_sync_conflict == "newer_wins"
        screen.query_one("#library-notes-sync-conflict").press()
        await pilot.pause()
        assert screen._library_notes_sync_conflict == "disk_wins"
        from tldw_chatbook.config import get_cli_setting
        assert get_cli_setting("notes", "sync_conflict_resolution", None) == "disk_wins"


@pytest.mark.asyncio
async def test_library_shell_notes_sync_auto_toggle_flips_and_persists():
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        assert screen._library_notes_sync_auto is False
        toggle = screen.query_one("#library-notes-sync-auto", Button)
        assert "○" in str(toggle.label)

        toggle.press()
        await pilot.pause()
        assert screen._library_notes_sync_auto is True
        assert screen._library_notes_auto_sync_timer is not None
        from tldw_chatbook.config import get_cli_setting
        assert get_cli_setting("notes", "auto_sync", None) is True
        toggle = screen.query_one("#library-notes-sync-auto", Button)
        assert "✓" in str(toggle.label)

        toggle.press()
        await pilot.pause()
        assert screen._library_notes_sync_auto is False
        assert screen._library_notes_auto_sync_timer is None
        assert get_cli_setting("notes", "auto_sync", None) is False


@pytest.mark.asyncio
async def test_library_shell_notes_sync_now_calls_recording_service_with_chosen_enums(
    monkeypatch, tmp_path
):
    """Sync-now with a valid folder must call the sync seam with the chosen
    direction/conflict enums, and update status/activity without ever
    recomposing (same Static widget instance) mid-run."""
    from tldw_chatbook.Notes.sync_engine import ConflictResolution, SyncDirection
    from tldw_chatbook.Notes import sync_service as sync_service_module

    _RecordingNotesSyncService.instances.clear()
    monkeypatch.setattr(sync_service_module, "NotesSyncService", _RecordingNotesSyncService)

    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        folder_input = screen.query_one("#library-notes-sync-folder", Input)
        folder_input.value = str(tmp_path)
        folder_input.focus()
        await pilot.pause()

        # Cycle direction/conflict once each so the recorded call proves the
        # *chosen* enums (not just the defaults) are threaded through.
        screen.query_one("#library-notes-sync-direction").press()
        await pilot.pause()
        screen.query_one("#library-notes-sync-conflict").press()
        await pilot.pause()

        screen.query_one("#library-notes-sync-run").press()
        for _ in range(150):
            if screen._library_notes_sync_running:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync-now never started.")

        # Captured only once the start-of-sync recompose has already
        # happened, so this is the one-and-only Static instance the
        # progress callback's targeted update (and the finish handler)
        # must reuse -- proving no recompose happens mid-run.
        status_widget_before = screen.query_one("#library-notes-sync-status", Static)

        for _ in range(150):
            if _RecordingNotesSyncService.instances and _RecordingNotesSyncService.instances[0].calls:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync-now never reached the recording service.")

        service = _RecordingNotesSyncService.instances[0]
        call = service.calls[0]
        assert call["root_folder"] == tmp_path
        assert call["direction"] == SyncDirection.DISK_TO_DB
        assert call["conflict_resolution"] == ConflictResolution.DISK_WINS

        for _ in range(150):
            if service.progress_fired.is_set():
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Progress callback never fired.")
        for _ in range(150):
            if "syncing" in str(
                screen.query_one("#library-notes-sync-status", Static).renderable
            ):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Progress callback never landed on the status widget.")

        # The mid-run progress update must have landed on the very same
        # Static instance captured before the callback fired -- no
        # recompose happened in between.
        status_widget_mid_run = screen.query_one("#library-notes-sync-status", Static)
        assert status_widget_before is status_widget_mid_run
        assert "1/2" in str(status_widget_mid_run.renderable)

        # Now let the gated fake service return so the run completes.
        service.release_event.set()
        for _ in range(150):
            if not screen._library_notes_sync_running:
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Sync run never completed.")

        status_widget_after = screen.query_one("#library-notes-sync-status", Static)
        assert "done" in str(status_widget_after.renderable)


@pytest.mark.asyncio
async def test_library_shell_notes_sync_invalid_folder_notifies_quietly_no_run(monkeypatch):
    from tldw_chatbook.Notes import sync_service as sync_service_module

    _RecordingNotesSyncService.instances.clear()
    monkeypatch.setattr(sync_service_module, "NotesSyncService", _RecordingNotesSyncService)

    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    notifications = []
    app.notify = lambda message, **kwargs: notifications.append((message, kwargs))
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        folder_input = screen.query_one("#library-notes-sync-folder", Input)
        folder_input.value = "/definitely/does/not/exist/anywhere"
        folder_input.focus()
        await pilot.pause()

        screen.query_one("#library-notes-sync-run").press()
        await pilot.pause()
        await pilot.pause()

        assert not _RecordingNotesSyncService.instances or not _RecordingNotesSyncService.instances[0].calls
        assert notifications
        assert notifications[-1][1].get("severity") == "warning"


@pytest.mark.asyncio
async def test_library_shell_notes_sync_rail_reentry_resets_transient_state():
    """Leaving the sync panel for another rail row and returning to Browse >
    Notes must clear stale status/activity from the previous visit."""
    app = _build_test_app()
    _prepare_library_notes_sync_app(app)
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_notes_sync_panel(screen, pilot)

        screen._library_notes_sync_status = "failed · disk full"
        screen._library_notes_sync_activity = ("some previous line",)

        screen.query_one("#library-row-browse-media").press()
        await pilot.pause()
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-sync-open")

        assert screen._library_notes_sync_status == "idle"
        assert screen._library_notes_sync_activity == ()
        assert screen._library_notes_view == "list"
