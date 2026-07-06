"""Library shell (L1) rail + conversations canvas pilot contracts."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App

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


def _seed_conversations(app, conversations, *, notes=None, media=None):
    app.notes_scope_service = StaticLibraryNotesScopeService(notes or [])
    app.media_reading_scope_service = StaticLibraryMediaScopeService(media or [])
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
async def test_library_shell_media_row_navigates_without_selection_change():
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

        assert seen[-1] == "media"
        assert screen._library_selected_row_id == ""
        assert screen.query_one("#library-canvas-landing")


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
        assert stale not in generated_css, stale
