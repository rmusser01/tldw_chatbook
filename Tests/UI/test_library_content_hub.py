"""Library content hub mounted regressions."""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from typing import Any

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_MODE,
)

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_selector,
)


async def _wait_for_library_shell_ready(screen, pilot, *, timeout: float = 2.0) -> None:
    """Wait for the Library rail shell (not the retired Content Hub) to mount.

    Mirrors ``Tests/UI/test_library_shell.py::_wait_for_library_shell`` for
    suites that use the generic ``DestinationHarness``.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if getattr(screen, "_library_loaded", False) and screen.query("#library-rail"):
            await pilot.pause()
            await pilot.pause()
            return
        await pilot.pause(0.01)
    raise AssertionError(
        f"Library shell never loaded. Visible text: {_visible_text(screen)}"
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


async def _wait_for_library_conversation_selection(
    screen,
    pilot,
    conversation_id: str,
    expected_title: str,
    *,
    attempts: int = 80,
) -> None:
    for _ in range(attempts):
        if (
            getattr(screen, "_selected_conversation_id", None) == conversation_id
            and expected_title in _visible_text(screen)
        ):
            await pilot.pause()
            return
        await pilot.pause(0.05)
    raise AssertionError(
        f"Conversation {conversation_id!r} was not selected. "
        f"selected={getattr(screen, '_selected_conversation_id', None)!r}; "
        f"visible={_visible_text(screen)}"
    )


class StaticLibraryCollectionsService:
    """Small mounted-test service for Library Collections snapshots."""

    def __init__(self, records) -> None:
        self.records = tuple(records)

    def list_collections(self):
        return self.records


class StaticLibraryRagSearchService:
    """Mounted-test retrieval service for Library Search/RAG evidence rows."""

    def __init__(self, results: Iterable[object]) -> None:
        self.results = tuple(results)
        self.requests: list[dict[str, object]] = []

    async def search(
        self,
        query: str,
        scope: Sequence[str],
        mode: str,
        **kwargs: Any,
    ) -> dict[str, object]:
        """Record a Search/RAG request and return static evidence rows.

        Args:
            query: User-entered retrieval query.
            scope: Library source scopes included in the request.
            mode: Retrieval mode requested by the UI.
            **kwargs: Additional request metadata forwarded by the screen.

        Returns:
            A deterministic service payload containing the fake backend label and
            the preconfigured result rows.
        """
        self.requests.append(
            {
                "query": query,
                "scope": tuple(scope),
                "mode": mode,
                "kwargs": dict(kwargs),
            }
        )
        return {
            "runtime_backend": "local-test",
            "results": self.results,
        }




@pytest.mark.asyncio
async def test_library_stage_c_search_rag_promotes_query_scope_and_evidence_regions() -> None:
    """The Search/RAG mode canvas (``LibrarySearchRagPanel``) still promotes
    query, scope, and evidence regions. The dedicated Console-handoff/
    inspector digest ("Console Handoff", "Selected Evidence: none", "Future
    Attribution" headings) lived only in the retired 3-pane inspector column
    (``LibrarySearchRagInspectorPanel``, never mounted by the new canvas) and
    has no successor here; that Console-handoff decision is now covered by
    the in-panel per-result "Use in Console" button (see the sibling
    selected-evidence test below).

    Re-anchored for the L3a UX wave (A1/A3/A4/B1/B2/B3): the idle canvas is
    now one quiet line instead of the ~9-line redundant blocked dump, ASCII
    section rules and the pipe-drawn scope table are gone in favor of
    Console-parity headings and real per-source toggles, and the carry-
    through jargon line (B3) is retired.
    """
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        visible = _visible_text(screen)

        # A1: exactly one quiet line for the empty-query gate; no callout,
        # no summary Static, no "Run disabled:" reason, no recovery dump.
        assert screen.query_one("#library-rag-query-input")
        assert screen.query_one("#library-rag-run-query", Button).disabled is True
        assert screen.query_one("#library-rag-query-quiet-line", Static)
        assert "Enter a question or search query." in visible
        assert not screen.query("#library-rag-query-blocked-callout")
        assert not screen.query("#library-rag-query-recovery")
        assert not screen.query("#library-rag-run-disabled-reason")
        assert "Blocked: enter a question or search query." not in visible
        assert "Blocked | Enter a question before running retrieval." not in visible
        assert "Run disabled: enter a question or search query." not in visible

        # A4: the retired-workbench shortcuts line is gone.
        assert not screen.query("#library-rag-query-shortcuts")
        assert "Tab: move panes" not in visible

        # B1: Console-parity section headers, no ASCII rules or duplicated
        # plain sub-headers.
        assert not screen.query(".library-rag-section-rule")
        assert "Retrieval Query" not in visible
        assert "Scope Controls" not in visible
        assert "Evidence Results" not in visible
        assert screen.query_one("#library-rag-scope-heading", Static)
        assert "Sources" in visible

        # B2: real per-source toggles replace the pipe-drawn scope table;
        # workspaces/collections/import-export rows are gone.
        assert not screen.query("#library-rag-scope-table-header")
        for source_type in ("notes", "media", "conversations"):
            toggle = screen.query_one(f"#library-rag-scope-toggle-{source_type}", Button)
            assert str(toggle.label).startswith("✓")
            assert toggle.disabled is False
        assert not screen.query("#library-rag-scope-row-all")
        assert not screen.query("#library-rag-scope-row-workspace")
        assert not screen.query("#library-rag-scope-row-collections")
        assert not screen.query("#library-rag-scope-row-import-export")
        assert "Workspace eligible" not in visible
        assert "Import/Export recovery" not in visible

        # A3: top-k surfaces on the Evidence heading, the single mode
        # surface is the toggle button, not a separate status line.
        assert "Evidence · top 5 per source" in visible
        assert not screen.query("#library-rag-query-status")
        assert "No evidence yet. Run Search/RAG to populate results." in visible
        assert screen.query_one("#library-rag-evidence-empty-guidance", Static)
        assert "Add or import sources, run a query, then select evidence for Console." in visible

        # B3: the carry-through jargon line is retired outright.
        assert not screen.query("#library-rag-attribution-placeholder")
        assert "Citation/snippet carry-through" not in visible
        assert "tldw_server" not in visible


@pytest.mark.asyncio
async def test_library_stage_c_search_rag_selected_evidence_updates_inspector_contract() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        [
            {
                "title": "Research Note",
                "snippet": "Useful answer evidence from the selected note.",
                "source_id": "note-1",
                "chunk_id": "chunk-7",
                "score": 0.82,
                "citations": [{"label": "Research Note #7", "source_id": "note-1"}],
                "provenance": {
                    "source_type": "notes",
                    "workspace_ids": ["default"],
                    "active_workspace_id": "default",
                    "active_context_eligible": True,
                    "authority_label": "local note",
                },
            }
        ]
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = "What does the research note say?"
        await screen.update_library_rag_query(Input.Changed(query_input, query_input.value))
        await _wait_for_selector(screen, pilot, "#library-rag-run-query")
        await screen._start_library_rag_query()
        await _wait_for_selector(screen, pilot, "#library-rag-select-result-0")
        await pilot.click("#library-rag-select-result-0")
        await _wait_for_selector(screen, pilot, "#library-rag-use-selected-in-console")

        visible = _visible_text(screen)

        # The dedicated retrieval-status/handoff-decision digest ("Retrieval
        # Status", "Use in Console: ready", "Allowed/Blocked actions") lived
        # in the retired inspector column; the panel itself now surfaces
        # selection, evidence, and Console eligibility directly.
        assert screen.query_one("#library-rag-result-0").has_class("is-selected")
        assert str(screen.query_one("#library-rag-select-result-0", Button).label) == "Selected evidence"
        # B3: the carry-through jargon line is retired outright -- selecting
        # evidence needs no permanent caption.
        assert not screen.query("#library-rag-attribution-placeholder")
        assert "Citation/snippet carry-through" not in visible
        assert "Useful answer evidence from the selected note." in visible
        assert "Citations: Research Note #7" in visible
        assert screen.query_one("#library-rag-use-selected-in-console", Button).disabled is False


@pytest.mark.asyncio
async def test_library_source_rail_marks_active_mode_without_mutating_action_labels() -> None:
    """Selecting a rail row marks it active (``library-rail-row-selected`` +
    a ``▸`` marker prefix) without mutating the row's underlying title."""
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        collections_row = screen.query_one("#library-row-browse-collections", Button)
        collections_row.press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        collections_row = screen.query_one("#library-row-browse-collections", Button)
        assert collections_row.has_class("library-rail-row-selected")
        assert str(collections_row.label).startswith("▸ Collections")
        # The row's underlying title (used for the tooltip) is unmutated by selection.
        assert collections_row.tooltip == "Collections"


@pytest.mark.asyncio
async def test_library_navigation_context_opens_requested_conversation() -> None:
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
        screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_CONVERSATION_ID: "chat-2"})
        await _wait_for_library_conversation_selection(
            screen,
            pilot,
            "chat-2",
            "Design Review",
        )

        visible = _visible_text(screen)
        assert getattr(screen, "_library_selected_row_id") == "browse-conversations"
        assert getattr(screen, "_selected_conversation_id") == "chat-2"
        assert "Design Review" in visible
        assert "Planning Chat" in visible
        assert screen.query_one("#library-row-browse-conversations", Button).has_class(
            "library-rail-row-selected"
        )


@pytest.mark.asyncio
async def test_library_navigation_context_opens_requested_valid_mode() -> None:
    app = _build_test_app()
    _seed_library_content(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_MODE: "search"})
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert getattr(screen, "_library_selected_row_id") == "browse-search"
        assert str(screen.query_one("#library-header-line").renderable) == "Library | Local"
        assert screen.query_one("#library-row-browse-search", Button).has_class(
            "library-rail-row-selected"
        )


@pytest.mark.asyncio
async def test_library_conversations_empty_state_is_honest_and_blocks_actions() -> None:
    """With no saved conversations, the canvas shows the honest empty copy
    and offers no Console-handoff affordance (the dedicated empty-state
    "Open Console" button and the "Use as source" action were dropped when
    the 3-pane Conversations mode was replaced by ``LibraryConversationsCanvas``;
    there is no live successor for either)."""
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        screen.query_one("#library-row-browse-conversations", Button).press()
        await _wait_for_selector(screen, pilot, "#library-conversations-status")

        status = str(screen.query_one("#library-conversations-status").renderable)
        assert status == "No conversations yet. Chat in Console and it appears here."
        assert not screen.query(".library-conversation-row")
        assert screen.query_one("#library-conversation-preview").display is False


@pytest.mark.asyncio
async def test_library_conversations_snapshot_requests_all_scopes() -> None:
    """The Library conversations snapshot must span workspace-scoped rows.

    Console chats persisted inside a workspace session are stored with
    ``scope_type='workspace'``; the service's default scope is 'global',
    which made Library Browse ▸ Conversations show "(0)" while the Console
    rail listed the same chats. The screen must explicitly request
    ``scope_type='all'``.
    """
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    conversation_service = StaticLibraryConversationScopeService(
        [
            {
                "title": "Console workspace chat",
                "conversation_id": "chat-ws-1",
                "message_count": 4,
                "workspace_id": "ws-chats",
                "updated_at": "2026-07-01T10:00:00Z",
            }
        ]
    )
    app.chat_conversation_scope_service = conversation_service
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        assert conversation_service.calls, "Library never fetched conversations"
        for call in conversation_service.calls:
            assert call.get("scope_type") == "all"

        screen.query_one("#library-row-browse-conversations", Button).press()
        await _wait_for_library_conversation_selection(
            screen,
            pilot,
            "chat-ws-1",
            "Console workspace chat",
        )
        assert "Console workspace chat" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_collections_selection_explains_membership_workspace_and_actions() -> None:
    """``LibraryCollectionsPanel`` (mounted verbatim in the canvas) still
    explains membership, workspace rule, and action status for a selected
    Collection. The retired 3-pane inspector column duplicated this same
    copy under different ids (``library-collection-inspector-*``) with a few
    inspector-only lines ("Selected Collection Record" heading, "Collection
    item reader: not wired locally yet.", the two "Disabled: collection item
    ... is not wired yet." lines, and the Search/RAG recovery sentence) that
    have no successor in the single-pane canvas."""
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
        await _wait_for_library_shell_ready(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-panel")

        visible = _visible_text(screen)

        assert "Launch Evidence" in visible
        assert "Stored item count: 3 items" in visible
        assert "Stored collection content" in visible
        assert "Selected: Launch Evidence" in visible
        assert "Available now: create, rename, delete records" in visible
        assert (
            "Blocked later: item reader, Search/RAG, Study, Console handoff, server sync"
        ) in visible
        assert "Next: collection item adapters are required before item-level actions unlock." in visible
        assert screen.query_one("#library-row-browse-collections", Button).has_class(
            "library-rail-row-selected"
        )
        assert screen.query_one("#library-use-in-console", Button).disabled is True


@pytest.mark.asyncio
async def test_library_collections_empty_state_keeps_global_browse_rule_and_blocks_wip_actions() -> None:
    """``LibraryCollectionsPanel``'s empty branch still teaches content entry
    and keeps the create/rename/delete form inert until a name is entered.
    The dead-inspector-only lines dropped here mirror the sibling selection
    test above (no live successor exists for them)."""
    app = _build_test_app()
    _seed_library_content(app)
    app.library_collections_service = StaticLibraryCollectionsService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        screen.query_one("#library-row-browse-collections", Button).press()
        await _wait_for_selector(screen, pilot, "#library-collections-empty")

        visible = _visible_text(screen)

        assert "No Collections yet." in visible
        assert "Create a local Collection record to start reviewing saved content." in visible
        assert "Type a Collection name to enable Create." in visible
        assert "Form actions: enter a name to enable Create." in visible
        assert "Create, Rename, and Delete stay inactive until their requirements are met." in visible
        assert "No stored collection items are available locally yet." in visible
        assert "Collections are for reading, reviewing, and reusing saved content." in visible
        assert "No Collection selected." in visible
        empty_reader = screen.query_one("#library-collection-empty-reader", Static)
        form_guidance = screen.query_one("#library-collection-form-guidance", Static)
        form_action_state = screen.query_one("#library-collection-form-action-state", Static)
        assert empty_reader.region.y <= form_guidance.region.y + 8
        assert form_action_state.region.y < screen.query_one("#library-create-collection", Button).region.y
        assert not screen.query("#library-collections-workbench")
        assert screen.query_one("#library-use-in-console", Button).disabled is True
