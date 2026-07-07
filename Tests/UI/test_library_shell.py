"""Library shell (L1) rail + conversations canvas pilot contracts."""

import re
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App
from textual.widgets import Button, Input, Static, TextArea

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.test_destination_shells import (
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
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
        assert "yellow" in visible
        assert screen.query_one("#library-media-highlight-delete-0")


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
        await _wait_for_selector(screen, pilot, "#library-media-highlight-quote")

        assert not screen.query(".library-media-highlight-delete")

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
