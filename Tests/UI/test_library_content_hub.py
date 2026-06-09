"""Library content hub mounted regressions."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.widgets import Button

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _link_library_items_to_active_workspace,
    _visible_text,
    _wait_for_library_snapshot,
    _wait_for_selector,
)


def _seed_library_content(app) -> None:
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "media_id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "conversation_id": "chat-1"}]
    )


class StaticLibraryCollectionsService:
    """Small mounted-test service for Library Collections snapshots."""

    def __init__(self, records) -> None:
        self.records = tuple(records)

    def list_collections(self):
        return self.records


@pytest.mark.asyncio
async def test_library_default_mode_renders_content_hub_with_real_counts_and_recent_items() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-content-hub-title")

        visible = _visible_text(screen)
        assert "Library Content Hub" in visible
        assert "Landing page for ingested content, notes, media, conversations, collections, imports/exports, and retrieval." in visible
        assert "Notes: 1" in visible
        assert "Recent: Research Note" in visible
        assert "Media: 1" in visible
        assert "Recent: Transcript A" in visible
        assert "Conversations: 1" in visible
        assert "Recent: Planning Chat" in visible
        assert "Search/RAG: query indexed Library content" in visible
        assert "Collections: organize reusable content groups inside Library" in visible
        assert "Study: turn Library content into flashcards and quizzes" in visible
        assert "Console handoff is secondary" in visible


@pytest.mark.asyncio
async def test_library_hub_recent_titles_render_rich_markup_literals() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "[bold]Literal Note[/]", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-content-hub-title")

        visible = _visible_text(screen)
        card = screen.query_one("#library-notes-summary")

        assert getattr(card, "_render_markup") is False
        assert "Recent: [bold]Literal Note[/]" in visible
        assert "Recent: Literal Note" not in visible


@pytest.mark.asyncio
async def test_library_hub_empty_state_teaches_content_entry_without_selected_source_actions() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(160, 44)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-source-empty")

        visible = _visible_text(screen)
        assert "No local Library content yet." in visible
        assert "Import media, create notes, or open Library Search/RAG after indexing." in visible
        assert "Library remains a hub; Notes, Media, Search/RAG, and Study own deeper work." in visible
        assert not screen.query("#library-selected-source-title")
        assert not screen.query("#library-selected-source-actions")


@pytest.mark.asyncio
async def test_library_hub_owner_route_actions_remain_keyboard_reachable() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        expected_owner_actions = {
            "#library-open-notes": "Open Notes",
            "#library-open-media": "Open Media",
            "#library-open-conversations": "Open Conversations",
            "#library-open-import-export": "Import/Export Sources",
            "#library-open-search": "Search/RAG",
            "#library-open-collections": "Collections",
        }
        for selector, label in expected_owner_actions.items():
            button = screen.query_one(selector, Button)
            assert str(button.label) == label
            assert button.disabled is False
        handoff_button = screen.query_one("#library-use-in-console", Button)
        assert str(handoff_button.label) == "Use in Console"
        assert "Console" in str(handoff_button.tooltip)

        visible = _visible_text(screen)
        assert "Notes owner: Notes screen handles editing, sync, templates, export, and delete." in visible
        assert "Media owner: Media screen handles browse, ingest review, analysis, and read-it-later." in visible
        assert "Search/RAG owner: Library Search/RAG handles retrieval, evidence, and saved searches." in visible


@pytest.mark.asyncio
async def test_library_hub_module_actions_are_visually_separated() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        action_buttons = [
            screen.query_one("#library-open-notes", Button),
            screen.query_one("#library-open-media", Button),
            screen.query_one("#library-open-conversations", Button),
            screen.query_one("#library-open-import-export", Button),
            screen.query_one("#library-open-search", Button),
            screen.query_one("#library-open-collections", Button),
        ]

        for button in action_buttons:
            assert button.region.height >= 3

        for previous, current in zip(action_buttons, action_buttons[1:]):
            assert current.region.y > previous.region.y + previous.region.height


@pytest.mark.asyncio
async def test_library_conversations_action_opens_native_browser_without_legacy_exit() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-conversations")
        await _wait_for_selector(screen, pilot, "#library-conversations-browser-title")

        visible = _visible_text(screen)
        assert _active_destination_screen(host) is screen
        assert "Conversations mode" in visible
        assert "Saved Conversations" in visible
        assert "Planning Chat" in visible
        assert screen.query_one("#library-conversation-select-0", Button).disabled is False
        assert "Library | Conversations | Ready | Local" in visible
        assert screen.query_one("#library-open-conversations", Button).has_class("is-active")


@pytest.mark.asyncio
async def test_library_conversations_selection_shows_metadata_and_handoff_actions() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [
            {
                "title": "Planning Chat",
                "conversation_id": "chat-1",
                "message_count": 7,
                "updated_at": "2026-06-01T10:00:00Z",
            },
            {
                "title": "Design Review",
                "conversation_id": "chat-2",
                "message_count": 3,
                "workspace_id": "ws-other",
                "last_modified": "2026-06-02T09:30:00Z",
            },
        ]
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-conversations")
        await _wait_for_selector(screen, pilot, "#library-conversation-select-1")
        await pilot.click("#library-conversation-select-1")
        await _wait_for_selector(screen, pilot, "#library-selected-conversation-title")

        visible = _visible_text(screen)
        open_button = screen.query_one("#library-conversation-open-console", Button)
        source_button = screen.query_one("#library-conversation-use-source", Button)

        assert "Design Review" in visible
        assert "Messages: 3" in visible
        assert "Source authority: local" in visible
        assert "Workspace: ws-other" in visible
        assert "Updated: 2026-06-02T09:30:00Z" in visible
        assert "Handoff eligibility:" in visible
        assert str(open_button.label) == "Open in Console"
        assert str(source_button.label) == "Use as source"


@pytest.mark.asyncio
async def test_library_conversations_empty_state_is_honest_and_blocks_actions() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-conversations")
        await _wait_for_selector(screen, pilot, "#library-conversations-empty")

        visible = _visible_text(screen)
        assert "Library | Conversations | Empty | Local" in visible
        assert "No saved conversations available in Library." in visible
        assert "Create or save a Console chat, then return here to browse it." in visible
        assert "What appears here:" in visible
        assert "Saved chats with title, message count, workspace, and updated time." in visible
        assert "Select a row to enable Console handoff actions." in visible
        assert "Select a conversation first to enable these actions." in visible
        open_console_button = screen.query_one("#library-conversations-open-console-empty", Button)
        assert str(open_console_button.label) == "Open Console"
        assert open_console_button.disabled is False
        assert screen.query_one("#library-conversation-open-console", Button).disabled is True
        assert screen.query_one("#library-conversation-use-source", Button).disabled is True


@pytest.mark.asyncio
async def test_library_conversation_use_as_source_hands_off_selected_conversation() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [
            {
                "title": "Planning Chat",
                "conversation_id": "chat-1",
                "message_count": 7,
                "last_modified": "2026-06-03T10:15:00Z",
            },
        ]
    )
    app.open_chat_with_handoff = Mock()
    _link_library_items_to_active_workspace(
        app,
        (("conversation", "chat-1", "Planning Chat"),),
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-conversations")
        await _wait_for_selector(screen, pilot, "#library-conversation-use-source")
        use_button = screen.query_one("#library-conversation-use-source", Button)

        assert use_button.disabled is False
        await pilot.click("#library-conversation-use-source")
        await pilot.pause(0.1)

    app.open_chat_with_handoff.assert_called_once()
    payload = app.open_chat_with_handoff.call_args.args[0]
    assert payload.source == "library"
    assert payload.item_type == "conversation"
    assert payload.source_id == "chat-1"
    assert payload.title == "Planning Chat"
    assert "Planning Chat" in payload.body
    assert "Updated: 2026-06-03T10:15:00Z" in payload.body
    assert payload.metadata["conversation_id"] == "chat-1"
    assert payload.metadata["updated_label"] == "Updated: 2026-06-03T10:15:00Z"


@pytest.mark.asyncio
async def test_library_import_export_opens_native_workflow_with_clear_boundaries() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    seen_routes: list[str] = []
    host = DestinationHarness(app, "library", seen_routes)

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-import-export")
        await _wait_for_selector(screen, pilot, "#library-import-export-workflow-title")

        visible = _visible_text(screen)
        assert getattr(screen, "_active_mode") == "import-export"
        assert seen_routes == []
        assert "Library | Import/Export | Ready | Local" in visible
        assert "Library Import/Export Workflow" in visible
        assert "Library owns source acquisition framing; Ingest and Media own deeper file handling." in visible
        assert "Import source material" in visible
        assert "Imported material returns here as notes, media, conversations, or indexed sources." in visible
        assert "Full Media ingestion and review stays in Media." in visible
        assert "Artifact export stays in Artifacts." in visible
        assert "Generic file management stays outside Library." in visible
        assert "Export is not wired here yet." in visible
        assert "Return path: come back to Library after import to see new hub inventory." in visible
        assert screen.query_one("#library-open-import-export", Button).has_class("is-active")
        assert screen.query_one("#library-import-export-open-ingest", Button).disabled is False
        assert screen.query_one("#library-import-export-open-media", Button).disabled is False
        assert screen.query_one("#library-import-export-export-sources", Button).disabled is True


@pytest.mark.asyncio
async def test_library_import_export_dedicated_actions_emit_handoff_routes() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    seen_routes: list[str] = []
    host = DestinationHarness(app, "library", seen_routes)

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-import-export")
        await _wait_for_selector(screen, pilot, "#library-import-export-open-ingest")

        await pilot.click("#library-import-export-open-ingest")
        await pilot.pause(0.1)
        await pilot.click("#library-import-export-open-media")
        await pilot.pause(0.1)

    assert seen_routes == ["ingest", "media"]


@pytest.mark.asyncio
async def test_library_collections_selection_explains_membership_workspace_and_actions() -> None:
    app = _build_test_app()
    _seed_library_content(app)
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
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-collections")
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        visible = _visible_text(screen)

        assert "Library | Collections | Ready | Local" in visible
        assert "Launch Evidence" in visible
        assert "Source membership" in visible
        assert "Membership: 3 items" in visible
        assert "Workspace boundary" in visible
        assert "Visible globally; active workspace controls staging and manipulation." in visible
        assert "Available now: create, rename, delete local Collection metadata." in visible
        assert (
            "Deferred: collection-scoped Search/RAG, Study, Console handoff, "
            "and server sync promotion."
        ) in visible
        assert (
            "Workspace rule: Library browsing/search stays global; "
            "Console/RAG staging follows active workspace."
        ) in visible
        assert "Blocked: collection-scoped Console handoff is not wired yet." in visible
        assert (
            "Recovery: use the Collection for local organization, or stage "
            "individual eligible sources from Library."
        ) in visible
        assert screen.query_one("#library-open-collections", Button).has_class("is-active")
        assert screen.query_one("#library-open-study", Button).disabled is True
        assert screen.query_one("#library-use-in-console", Button).disabled is True


@pytest.mark.asyncio
async def test_library_collections_empty_state_keeps_global_browse_rule_and_blocks_wip_actions() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    app.library_collections_service = StaticLibraryCollectionsService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-collections")
        await _wait_for_selector(screen, pilot, "#library-collections-empty")

        visible = _visible_text(screen)

        assert "Library | Collections | Empty | Local" in visible
        assert "Group saved Library items for Search/RAG, Study, and Console." in visible
        assert "No Collection selected." in visible
        assert (
            "Global browsing remains available; Collections only gate active "
            "staging and manipulation."
        ) in visible
        assert "Local actions available after creation: rename and delete Collection metadata." in visible
        assert (
            "WIP actions unavailable: collection-scoped Search/RAG, Study, "
            "Flashcards, Quizzes, Console handoff, and server sync promotion."
        ) in visible
        assert "Create a local Collection first, then select it to inspect membership." in visible
        assert screen.query_one("#library-open-study", Button).disabled is True
        assert screen.query_one("#library-use-in-console", Button).disabled is True
