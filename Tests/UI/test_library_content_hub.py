"""Library content hub mounted regressions."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesListScopeService,
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
        state_summary = str(screen.query_one("#library-hub-state-summary", Static).renderable)
        hub_header = str(screen.query_one("#library-content-hub-table-header", Static).renderable)
        notes_row = str(screen.query_one("#library-notes-summary", Static).renderable)
        media_row = str(screen.query_one("#library-media-summary", Static).renderable)
        conversations_row = str(
            screen.query_one("#library-conversations-summary", Static).renderable
        )

        assert "Library Content Hub" in visible
        assert "Source overview, retrieval readiness, movement paths, and next action." in visible
        assert "Content Hub mode" not in visible
        assert "Open the owning module for deep work" not in visible
        assert "State: Local workspace | Browse all workspaces | Console staging blocked" in state_summary
        assert "Inventory: Notes 1 | Media 1 | Conversations 1 | Console eligible 0 | Blocked 3" in state_summary
        assert "+-----------------+" in hub_header
        assert "| Source" in hub_header
        assert "| Count" in hub_header
        assert "| Browse" in hub_header
        assert "| Recent" in hub_header
        assert "| Console" in hub_header
        assert "Notes" in notes_row
        assert "1" in notes_row
        assert "Open Notes" in notes_row
        assert "Research Note" in notes_row
        assert "blocked: workspace gate" in notes_row
        assert notes_row.startswith("| Notes")
        assert "Notes: 1" in visible
        assert "Owners: Notes edits/sync/export" in visible
        assert "Media" in media_row
        assert "Transcript A" in media_row
        assert "blocked: workspace gate" in media_row
        assert "Media: 1" in visible
        assert "Media browses library items" in visible
        assert "Conversations" in conversations_row
        assert "Planning Chat" in conversations_row
        assert "blocked: workspace gate" in conversations_row
        assert "Conversations: 1" in visible
        assert "Conversations resumes chats" in visible
        assert "Library readiness" in visible
        assert "Eligible       0 modules" in visible
        assert "Blocked        3 workspace-gated modules" in visible
        assert "Recent         Notes: Research Note; Media: Transcript A; Conversations: Planning Chat" in visible
        assert "Next           Link sources to the active workspace or open an owner screen." in visible
        assert "Search/RAG     Query indexed content, inspect evidence, launch Console." in visible
        assert "Collections    Read, review, reuse saved content." in visible
        assert "Study          Turn Library content into flashcards and quizzes." in visible
        assert "Console handoff is secondary" in visible


@pytest.mark.asyncio
async def test_library_stage_a_shell_surfaces_source_map_workspace_context_and_inspector_actions() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-content-hub-title")

        visible = _visible_text(screen)

        assert "Source Map" in visible
        assert "Workspace Context" in visible
        assert "Quick Actions" in visible
        assert "Open a mode, then use the inspector for selected-item actions." in visible
        assert "Next action" in visible
        assert "Active Workbench" in visible
        assert "Browse: all workspaces" in visible
        assert "Use/stage: active workspace only" in visible
        assert "Browse/search: all workspaces" not in visible
        assert "Sources" in visible
        assert "Retrieval" in visible
        assert "Movement" in visible
        assert "Learning" in visible


@pytest.mark.asyncio
async def test_library_hub_inspector_prioritizes_selected_available_blocked_and_next() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-hub-actions-title")

        visible = _visible_text(screen)

        assert "Selected" in visible
        assert "Selected: none" in visible
        assert "Available now" in visible
        assert "Blocked" in visible
        assert "Next action" in visible
        assert "Details" in visible
        assert "Knowledge workflow" not in visible
        assert "Select a source module on the left to inspect browse and handoff rules." not in visible
        inspector = screen.query_one("#library-source-inspector")
        assert not inspector.query("#library-open-study")
        assert screen.query_one("#library-use-in-console", Button).disabled is True


@pytest.mark.asyncio
async def test_library_hub_detail_uses_scannable_sections_instead_of_report_copy() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-content-hub-title")

        visible = _visible_text(screen)

        for selector in (
            "#library-hub-section-source-status",
            "#library-hub-section-retrieval-readiness",
            "#library-hub-section-movement-reuse",
            "#library-hub-section-learning-paths",
            "#library-hub-section-next-action",
            "#library-hub-source-table-bottom",
            "#library-hub-spacer-after-source-table",
            "#library-hub-spacer-before-retrieval",
            "#library-hub-spacer-before-movement",
            "#library-hub-spacer-before-learning",
            "#library-hub-spacer-before-next-action",
        ):
            assert screen.query_one(selector, Static)

        assert "-- Source Status " in visible
        assert "-- Retrieval Readiness " in visible
        assert "-- Movement + Reuse " in visible
        assert "-- Learning " in visible
        assert "-- Next Action " in visible
        assert "| Notes" in visible
        assert "+-----------------+---------+--------------------+----------------------------------+---------------------------+" in visible
        assert "Purpose:" not in str(screen.query_one("#library-notes-summary", Static).renderable)
        assert "Search/RAG     Query indexed content, inspect evidence, launch Console." in visible
        assert "Import/Export  Add or move content; imported material returns here." in visible
        assert "Collections    Read, review, reuse saved content." in visible
        assert "Study          Turn Library content into flashcards and quizzes." in visible
        assert "Primary        Import sources or create a note." in visible
        assert "Then           Open Search/RAG after indexing or Collections after saving content." in visible


@pytest.mark.asyncio
async def test_library_source_rail_marks_active_mode_without_mutating_action_labels() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await pilot.click("#library-open-collections")
        await _wait_for_selector(screen, pilot, "#library-source-active-marker")

        marker = screen.query_one("#library-source-active-marker", Static)
        collections_button = screen.query_one("#library-open-collections", Button)

        assert str(marker.renderable) == "> Active: Collections"
        assert str(collections_button.label) == "Collections"
        assert collections_button.has_class("is-active")


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
        readiness = screen.query_one("#library-hub-readiness-summary")

        assert getattr(card, "_render_markup") is False
        assert getattr(readiness, "_render_markup") is False
        assert "[bold]Literal Note[/]" in visible
        assert "blocked: workspace gate" in visible
        assert "Notes | Notes: 1 | Open Notes | Literal Note | blocked" not in visible


@pytest.mark.asyncio
async def test_library_hub_recent_column_uses_one_shortened_title() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesListScopeService(
        [
            {
                "title": "A very long note title that should not consume the whole content hub row",
                "id": "note-1",
            },
            {"title": "Second recent note should not appear in the row", "id": "note-2"},
        ]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-content-hub-title")

        visible = _visible_text(screen)

        assert "A very long note title that..." in visible
        assert "Second recent note should not appear" not in visible
        assert "showing up to" not in str(screen.query_one("#library-notes-summary", Static).renderable)


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
async def test_library_hub_module_actions_use_compact_console_rows() -> None:
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
            screen.query_one("#library-open-search", Button),
            screen.query_one("#library-open-import-export", Button),
            screen.query_one("#library-open-collections", Button),
        ]

        for button in action_buttons:
            assert button.region.height == 1

        for previous, current in zip(action_buttons, action_buttons[1:]):
            assert current.region.y >= previous.region.y + previous.region.height

        visible = _visible_text(screen)
        assert "Notes: 1 | global browse | stage gated" in visible
        assert "Media: 1 | global browse | stage gated" in visible
        assert "Conversations: 1 | global browse | stage gated" in visible
        assert "Retrieval | query first | stage evidence" in visible
        assert "Collections | read/review | items WIP" in visible


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
        assert "Collections Reader" in visible
        assert "Launch Evidence" in visible
        assert "Selected Collection Record" in visible
        assert "Stored item count: 3 items" in visible
        assert "Collection item reader: not wired locally yet." in visible
        assert "Stored collection content" in visible
        assert "Selected: Launch Evidence" in visible
        assert "Available now: create, rename, delete records" in visible
        assert "Item actions unavailable until collection items exist." in visible
        assert "Read/review collection items when a local item adapter is available." in visible
        assert (
            "Disabled: collection item Search/RAG, Study, Console handoff, and "
            "server sync promotion are not wired yet."
        ) in visible
        assert (
            "Blocked later: item reader, Search/RAG, Study, Console handoff, server sync"
        ) in visible
        assert "Next: collection item adapters are required before item-level actions unlock." in visible
        assert "Available now: create, rename, and delete local Collection records." not in visible
        assert (
            "Later: collection item reader, Search/RAG, Study, Console handoff, "
            "and server sync promotion."
        ) not in visible
        assert (
            "Workspace rule: Library browsing/search stays global; "
            "Console/RAG staging follows active workspace."
        ) in visible
        assert "Disabled: collection item Search/RAG is not wired yet." in visible
        assert "Disabled: collection item Console handoff is not wired yet." in visible
        assert (
            "Recovery: use existing Library Search/RAG or individual eligible sources "
            "until collection item adapters are available."
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
        mode_description = screen.query_one("#library-active-mode-description", Static)
        mode_next_action = screen.query_one("#library-active-mode-next-action", Static)

        assert "Library | Collections | Empty | Local" in visible
        assert mode_description.display is False
        assert mode_next_action.display is False
        assert "No Collections yet." in visible
        assert "Create a local Collection record to start reviewing saved content." in visible
        assert "Type a Collection name to enable Create." in visible
        assert "Form actions: enter a name to enable Create." in visible
        assert "Create, Rename, and Delete stay inactive until their requirements are met." in visible
        assert "No stored collection items are available locally yet." in visible
        assert "Collections are for reading, reviewing, and reusing saved content." in visible
        assert "No Collection selected." in visible
        assert (
            "Global browsing/search remains available; active staging and manipulation "
            "stay workspace-gated."
        ) in visible
        assert "Selected: none" in visible
        assert "Available now: create, rename, delete records" in visible
        assert "Item actions unavailable until collection items exist." in visible
        assert "Read/review collection items when a local item adapter is available." in visible
        assert (
            "Disabled: collection item Search/RAG, Study, Console handoff, and "
            "server sync promotion are not wired yet."
        ) in visible
        assert (
            "Blocked later: item reader, Search/RAG, Study, Console handoff, server sync"
        ) in visible
        assert visible.count("Blocked later:") == 1
        assert (
            "Next: select or create a Collection record to inspect local item-reader readiness."
        ) in visible
        assert "Available now: create, rename, and delete local Collection records." not in visible
        assert "Later: collection item reader, Search/RAG, Study, Console handoff, and server sync promotion." not in visible
        empty_reader = screen.query_one("#library-collection-empty-reader", Static)
        form_guidance = screen.query_one("#library-collection-form-guidance", Static)
        form_action_state = screen.query_one("#library-collection-form-action-state", Static)
        assert empty_reader.region.y <= form_guidance.region.y + 8
        assert form_action_state.region.y < screen.query_one("#library-create-collection", Button).region.y
        assert not screen.query("#library-collections-workbench")
        assert screen.query_one("#library-open-study", Button).disabled is True
        assert screen.query_one("#library-use-in-console", Button).disabled is True
