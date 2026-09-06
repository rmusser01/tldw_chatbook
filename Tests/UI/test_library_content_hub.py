"""Library content hub mounted regressions."""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Iterable, Sequence
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from textual.widgets import Button, Input, Static

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
from tldw_chatbook import Constants as constants
from tldw_chatbook.Character_Chat import (
    character_conversation_navigation as character_service_module,
)
from tldw_chatbook.Character_Chat.character_conversation_navigation import (
    CharacterConversationPage,
    CharacterConversationRow,
    UnavailableCharacterReason,
    UnresolvedConversationKey,
)
from tldw_chatbook.Constants import (
    LIBRARY_NAV_CONTEXT_CONVERSATION_ID,
    LIBRARY_NAV_CONTEXT_MODE,
)
from tldw_chatbook.UI.Navigation import (
    character_conversation_navigation as character_navigation,
)
from tldw_chatbook.UI.Screens import library_screen as library_screen_module


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
        if getattr(
            screen, "_selected_conversation_id", None
        ) == conversation_id and expected_title in _visible_text(screen):
            await pilot.pause()
            return
        await asyncio.sleep(0.05)
    raise AssertionError(
        f"Conversation {conversation_id!r} was not selected. "
        f"selected={getattr(screen, '_selected_conversation_id', None)!r}; "
        f"visible={_visible_text(screen)}"
    )


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
async def test_library_stage_c_search_rag_promotes_query_scope_and_evidence_regions() -> (
    None
):
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

        # task-2859 item 7: the canvas title drops the "Library " prefix
        # and matches the rail row's own "Search / RAG" spacing (it used
        # to read "Library Search/RAG", disagreeing with the rail on both
        # counts).
        assert (
            str(screen.query_one("#library-rag-panel-title", Static).renderable)
            == "Search / RAG"
        )
        assert "Library Search/RAG" not in visible

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
            toggle = screen.query_one(
                f"#library-rag-scope-toggle-{source_type}", Button
            )
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
        # TASK-15020/B3: that depth is the ACTIVE RAG PROFILE's
        # `search.default_top_k` -- 15 on the shipped default profile
        # (`hybrid_basic`), which is what an isolated test profile resolves
        # to -- not the old hardcoded 5.
        assert "Evidence · top 15 per source" in visible
        assert not screen.query("#library-rag-query-status")
        assert "No evidence yet. Run Search/RAG to populate results." in visible
        assert screen.query_one("#library-rag-evidence-empty-guidance", Static)
        assert (
            "Add or import sources, run a query, then select evidence for Console."
            in visible
        )

        # B3: the carry-through jargon line is retired outright.
        assert not screen.query("#library-rag-attribution-placeholder")
        assert "Citation/snippet carry-through" not in visible
        assert "tldw_server" not in visible


@pytest.mark.asyncio
async def test_library_stage_c_search_rag_selected_evidence_updates_inspector_contract() -> (
    None
):
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
        await screen.update_library_rag_query(
            Input.Changed(query_input, query_input.value)
        )
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
        assert (
            str(screen.query_one("#library-rag-select-result-0", Button).label)
            == "Selected evidence"
        )
        # task-2859 item 10: "N results for 'query'" headline above the
        # evidence cards -- there used to be no line naming the result
        # count or the query that produced it.
        assert (
            str(screen.query_one("#library-rag-results-count-line", Static).renderable)
            == "1 result for 'What does the research note say?'."
        )
        # B3: the carry-through jargon line is retired outright -- selecting
        # evidence needs no permanent caption.
        assert not screen.query("#library-rag-attribution-placeholder")
        assert "Citation/snippet carry-through" not in visible
        assert "Useful answer evidence from the selected note." in visible
        assert "Citations: Research Note #7" in visible
        assert (
            screen.query_one("#library-rag-use-selected-in-console", Button).disabled
            is False
        )
        # Note on the snippet-padding half of task-2859 item 10: this
        # harness (`DestinationHarness`) hosts the screen under a bare
        # `App` with no `CSS_PATH` of its own -- only widget-level
        # `DEFAULT_CSS` Python blocks are loaded, never the app bundle
        # (`css/tldw_cli_modular.tcss`), so `.library-rag-result-snippet`'s
        # new padding rule (bundle-only CSS) cannot be observed via
        # rendered geometry here. That check lives in
        # `Tests/UI/test_library_shell.py`, whose `LibraryHarness` sets
        # `CSS_PATH` to the real bundle for exactly this reason.


@pytest.mark.asyncio
async def test_library_source_rail_marks_active_mode_without_mutating_action_labels() -> (
    None
):
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
        await _wait_for_selector(screen, pilot, "#library-collections-reader-shell")

        collections_row = screen.query_one("#library-row-browse-collections", Button)
        assert collections_row.has_class("library-rail-row-selected")
        assert str(collections_row.label).startswith("▸ ")
        assert collections_row.library_row.title == "Collections"
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
async def test_library_unavailable_inspection_opens_exact_detail_once_without_repair() -> (
    None
):
    """Inspection keeps its typed intent through exact detail settlement."""

    assert hasattr(constants, "LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION")
    assert hasattr(constants, "LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE")
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [
            {"title": "Other chat", "conversation_id": "chat-1"},
            {"title": "Unavailable chat", "conversation_id": "chat-2"},
        ]
    )
    authority = "task4-library-authority"
    app.chachanotes_db = SimpleNamespace(
        get_local_authority_id=lambda: authority,
    )
    unresolved = UnresolvedConversationKey(authority, "chat-2")
    return_target = (
        character_navigation.RoleplayReturnTarget.console_context_character()
    )
    key = constants.LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION
    value = character_navigation.serialize_library_unavailable_inspection(
        character_navigation.LibraryUnavailableConversationInspection(
            unresolved, return_target
        )
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context({key: value})
        await _wait_for_library_conversation_selection(
            screen, pilot, "chat-2", "Unavailable chat"
        )

        assert not screen.query("#library-character-repair-dialog")
        assert screen._library_selected_row_id == "browse-conversations"
        assert screen._pending_library_source_open is None
        locator_calls = [
            call
            for call in app.chat_conversation_scope_service.calls
            if call.get("locator")
        ]
        assert len(locator_calls) == 1
        await screen._unavailable_navigation._open_pending_library_character_navigation(
            screen,
        )
        assert (
            len(
                [
                    call
                    for call in app.chat_conversation_scope_service.calls
                    if call.get("locator")
                ]
            )
            == 1
        )


@pytest.mark.asyncio
async def test_library_unavailable_browse_is_complete_explicit_projection(
    monkeypatch,
) -> None:
    """Browse excludes ordinary resolved rows and preserves the selected anchor."""

    authority = "task4-library-authority"
    unresolved_rows = tuple(
        CharacterConversationRow.unavailable(
            UnresolvedConversationKey(authority, conversation_id),
            reason=UnavailableCharacterReason.MISSING_CARD,
            character_label="Missing",
            title=title,
            last_modified=f"2026-09-0{index}T10:00:00Z",
            created_at="2026-09-01T00:00:00Z",
        )
        for index, (conversation_id, title) in enumerate(
            (("chat-2", "Unavailable chat"), ("chat-3", "Also unavailable")),
            start=1,
        )
    )

    class NavigationService:
        def __init__(self, _database):
            pass

        def unavailable_page(self, *, offset: int, limit: int):
            rows = unresolved_rows[offset : offset + limit]
            return CharacterConversationPage(rows, 2, None, 7)

    monkeypatch.setattr(
        character_service_module,
        "CharacterConversationNavigationService",
        NavigationService,
    )
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Ordinary resolved chat", "conversation_id": "chat-1"}]
    )
    app.chachanotes_db = SimpleNamespace(get_local_authority_id=lambda: authority)
    selected = UnresolvedConversationKey(authority, "chat-3")
    payload = character_navigation.serialize_library_unavailable_browse(
        character_navigation.LibraryUnavailableConversationsBrowse(
            selected,
            character_navigation.RoleplayReturnTarget.console_context_character(),
        )
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context(
            {constants.LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE: payload}
        )
        for _ in range(80):
            if screen._conversations_state.total == 2:
                break
            await pilot.pause(0.05)

        assert screen._conversations_state.total == 2
        assert screen._selected_conversation_id == "chat-3"
        assert {
            str(record.get("conversation_id"))
            for record in screen._conversations_state.page_records
        } == {"chat-2", "chat-3"}
        await _wait_for_selector(screen, pilot, "#library-conversations-title")
        title = screen.query_one("#library-conversations-title", Static)
        assert str(title.renderable) == "Unavailable character conversations (2)"
        assert "Ordinary resolved chat" not in _visible_text(screen)
        assert screen._pending_library_character_navigation is None


@pytest.mark.asyncio
async def test_library_unavailable_browse_owns_page_filter_and_retry(
    monkeypatch,
) -> None:
    """Pager/filter/retry must stay on the retained unavailable-only source."""

    authority = "task4-library-authority"
    page_size = library_screen_module.LIBRARY_CONVERSATION_PAGE_SIZE
    rows = tuple(
        CharacterConversationRow.unavailable(
            UnresolvedConversationKey(authority, f"chat-{index:02d}"),
            reason=UnavailableCharacterReason.MISSING_CARD,
            character_label="Missing",
            title=("Special unavailable" if index == page_size else f"Lost {index}"),
            last_modified=f"2026-09-03T10:{index:02d}:00Z",
            created_at="2026-09-01T00:00:00Z",
        )
        for index in range(page_size + 1)
    )

    class NavigationService:
        calls: ClassVar[list[tuple[int, int, str]]] = []
        fail_retry_once = True

        def __init__(self, _database):
            pass

        def unavailable_page(self, *, offset: int, limit: int, query: str = ""):
            type(self).calls.append((offset, limit, query))
            if query == "retry" and type(self).fail_retry_once:
                type(self).fail_retry_once = False
                raise RuntimeError("temporary failure")
            matching = tuple(
                row
                for row in rows
                if not query or query.casefold() in row.title.casefold()
            )
            return CharacterConversationPage(
                matching[offset : offset + limit],
                len(matching),
                None,
                7,
            )

    monkeypatch.setattr(
        character_service_module,
        "CharacterConversationNavigationService",
        NavigationService,
    )
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    ordinary = StaticLibraryConversationScopeService(
        [{"title": "Ordinary resolved chat", "conversation_id": "ordinary"}]
    )
    app.chat_conversation_scope_service = ordinary
    database = SimpleNamespace(get_local_authority_id=lambda: authority)
    app.chachanotes_db = database
    selected = UnresolvedConversationKey(authority, f"chat-{page_size:02d}")
    payload = character_navigation.serialize_library_unavailable_browse(
        character_navigation.LibraryUnavailableConversationsBrowse(
            selected,
            character_navigation.RoleplayReturnTarget.console_context_character(),
        )
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context(
            {constants.LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE: payload}
        )
        for _ in range(80):
            if screen._conversations_state.total == page_size + 1:
                break
            await pilot.pause(0.05)
        ordinary_calls = len(ordinary.calls)
        assert screen._pending_library_character_navigation is None
        assert screen._library_unavailable_browse_scope is not None

        screen.query_one("#library-conversations-next", Button).press()
        for _ in range(80):
            if screen._conversations_state.page == 2:
                break
            await pilot.pause(0.05)
        assert NavigationService.calls[-1] == (page_size, page_size, "")
        assert screen._conversations_state.total == page_size + 1
        assert [
            record["conversation_id"]
            for record in screen._conversations_state.page_records
        ] == [f"chat-{page_size:02d}"]
        assert screen._selected_conversation_id == selected.conversation_id
        assert len(ordinary.calls) == ordinary_calls

        filter_input = screen.query_one("#library-conversations-filter", Input)
        filter_input.value = "Special"
        filter_input.focus()
        await pilot.press("enter")
        expected_filter_call = (0, page_size, "Special")
        for _ in range(80):
            if (
                screen._conversations_state.requested_query == "Special"
                and not screen._conversations_state.loading
                and NavigationService.calls[-1:] == [expected_filter_call]
            ):
                break
            await pilot.pause(0.05)
        assert NavigationService.calls[-1] == expected_filter_call
        assert screen._conversations_state.total == 1
        assert len(ordinary.calls) == ordinary_calls

        filter_input = screen.query_one("#library-conversations-filter", Input)
        filter_input.value = "retry"
        filter_input.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-conversations-retry")
        screen.query_one("#library-conversations-retry", Button).press()
        expected_retry_calls = [
            (0, page_size, "retry"),
            (0, page_size, "retry"),
        ]
        for _ in range(80):
            if (
                screen._conversations_state.requested_query == "retry"
                and not screen._conversations_state.loading
                and not screen._conversations_state.error
                and NavigationService.calls[-2:] == expected_retry_calls
            ):
                break
            await pilot.pause(0.05)
        assert NavigationService.calls[-2:] == expected_retry_calls
        assert len(ordinary.calls) == ordinary_calls
        assert screen._library_unavailable_browse_scope is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ("service", "final_recompose"))
async def test_library_unavailable_browse_rejects_profile_churn_before_commit(
    monkeypatch,
    blocked_stage: str,
) -> None:
    """Old-profile unavailable rows never commit across either final await."""

    authority = "task4-library-authority"
    entered = threading.Event()
    release = threading.Event()

    class NavigationService:
        returned = False

        def __init__(self, _database):
            pass

        def unavailable_page(self, *, offset: int, limit: int, query: str = ""):
            if blocked_stage == "service":
                entered.set()
                assert release.wait(2)
            type(self).returned = True
            row = CharacterConversationRow.unavailable(
                UnresolvedConversationKey(authority, "old-unavailable"),
                reason=UnavailableCharacterReason.MISSING_CARD,
                character_label="Missing",
                title="Old unavailable",
                last_modified="2026-09-03T10:00:00Z",
                created_at="2026-09-01T00:00:00Z",
            )
            return CharacterConversationPage((row,), 1, None, 7)

    monkeypatch.setattr(
        character_service_module,
        "CharacterConversationNavigationService",
        NavigationService,
    )
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    database = SimpleNamespace(get_local_authority_id=lambda: authority)
    app.chachanotes_db = database
    selected = UnresolvedConversationKey(authority, "old-unavailable")
    payload = character_navigation.serialize_library_unavailable_browse(
        character_navigation.LibraryUnavailableConversationsBrowse(
            selected,
            character_navigation.RoleplayReturnTarget.console_context_character(),
        )
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        original_recompose = screen.recompose

        async def blocked_recompose():
            if (
                blocked_stage == "final_recompose"
                and NavigationService.returned
                and not entered.is_set()
            ):
                entered.set()
                assert await asyncio.to_thread(release.wait, 2)
            return await original_recompose()

        monkeypatch.setattr(screen, "recompose", blocked_recompose)
        screen.apply_navigation_context(
            {constants.LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE: payload}
        )
        assert await asyncio.to_thread(entered.wait, 2)
        app.chachanotes_db = SimpleNamespace(get_local_authority_id=lambda: authority)
        release.set()
        await pilot.pause(0.3)

        assert screen._pending_library_character_navigation is None
        assert screen._library_unavailable_browse_scope is None
        assert screen._conversations_state.loading is False
        assert all(
            record.get("conversation_id") != "old-unavailable"
            for record in screen._conversations_state.page_records
        )
        assert screen._selected_conversation_id != "old-unavailable"
        assert "Old unavailable" not in _visible_text(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("blocked_stage", ("flush", "lookup"))
async def test_library_character_route_rejects_profile_churn_across_await(
    blocked_stage: str,
) -> None:
    """A new DB handle cannot consume an admitted typed route, even with same authority."""

    authority = "task4-library-authority"
    entered = asyncio.Event()
    release = asyncio.Event()

    class BlockingConversationService(StaticLibraryConversationScopeService):
        async def locate_conversation_page(self, conversation_id, **kwargs):
            if blocked_stage == "lookup":
                entered.set()
                await release.wait()
            return await super().locate_conversation_page(conversation_id, **kwargs)

    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    service = BlockingConversationService(
        [{"title": "Unavailable chat", "conversation_id": "chat-2"}]
    )
    app.chat_conversation_scope_service = service
    old_database = SimpleNamespace(get_local_authority_id=lambda: authority)
    app.chachanotes_db = old_database
    unresolved = UnresolvedConversationKey(authority, "chat-2")
    payload = character_navigation.serialize_library_unavailable_inspection(
        character_navigation.LibraryUnavailableConversationInspection(
            unresolved,
            character_navigation.RoleplayReturnTarget.console_context_character(),
        )
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        if blocked_stage == "flush":

            async def blocked_flush():
                entered.set()
                await release.wait()
                return True

            screen._flush_active_file_notes = blocked_flush
        screen.apply_navigation_context(
            {constants.LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION: payload}
        )
        await asyncio.wait_for(entered.wait(), 2)
        app.chachanotes_db = SimpleNamespace(get_local_authority_id=lambda: authority)
        release.set()
        await pilot.pause(0.3)

        assert screen._selected_conversation_id != "chat-2"
        assert screen._pending_library_character_navigation is None
        if blocked_stage == "flush":
            assert not any(call.get("locator") for call in service.calls)


@pytest.mark.asyncio
async def test_library_character_route_rejects_authority_mismatch() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    service = StaticLibraryConversationScopeService(
        [{"title": "Wrong profile", "conversation_id": "chat-2"}]
    )
    app.chat_conversation_scope_service = service
    app.chachanotes_db = SimpleNamespace(
        get_local_authority_id=lambda: "active-authority"
    )
    payload = character_navigation.serialize_library_unavailable_inspection(
        character_navigation.LibraryUnavailableConversationInspection(
            UnresolvedConversationKey("other-authority", "chat-2"),
            character_navigation.RoleplayReturnTarget.console_context_character(),
        )
    )
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(180, 50)) as pilot:
        screen = _active_destination_screen(host)
        screen.apply_navigation_context(
            {constants.LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION: payload}
        )
        await pilot.pause(0.2)

        assert screen._selected_conversation_id != "chat-2"
        assert screen._pending_library_character_navigation is None
        assert not any(call.get("locator") for call in service.calls)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "incoming",
    (
        "invalid_repair",
        "invalid_character",
        "ordinary_veto",
        "typed_veto",
        "superseded_typed",
    ),
)
async def test_rejected_navigation_preserves_committed_unavailable_browse(
    monkeypatch, incoming
):
    from tldw_chatbook.UI.Navigation.main_navigation import NavigateToScreen

    authority = "retained-authority"
    key = UnresolvedConversationKey(authority, "retained-chat")
    row = CharacterConversationRow.unavailable(
        key,
        reason=UnavailableCharacterReason.MISSING_CARD,
        character_label="Unavailable",
        title="Retained chat",
        last_modified="2026-09-03T10:00:00Z",
        created_at="2026-09-01T00:00:00Z",
    )

    class NavigationService:
        def __init__(self, database):
            pass

        def unavailable_page(self, *, offset, limit, query=""):
            return CharacterConversationPage((row,), 1, None, 7)

    monkeypatch.setattr(
        character_service_module,
        "CharacterConversationNavigationService",
        NavigationService,
    )
    app = _build_test_app()
    _seed_library_content(app)
    app.chachanotes_db = SimpleNamespace(get_local_authority_id=lambda: authority)
    route = character_navigation.LibraryUnavailableConversationsBrowse(
        key, character_navigation.RoleplayReturnTarget.console_context_character()
    )
    context = {
        constants.LIBRARY_NAV_CONTEXT_CHARACTER_BROWSE: character_navigation.serialize_library_unavailable_browse(
            route
        )
    }
    returned = []
    host = DestinationHarness(app, "library")
    async with host.run_test(
        size=(120, 40),
        message_hook=lambda message: (
            returned.append(message) if isinstance(message, NavigateToScreen) else None
        ),
    ) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)
        screen.apply_navigation_context(context)
        for _ in range(60):
            await pilot.pause(0.05)
            if screen._conversations_state.page_loaded and screen.query(
                "#library-character-back-console"
            ):
                break
        back = screen.query_one("#library-character-back-console", Button)
        committed = screen._navigation_controller.character_route
        scope = screen._library_unavailable_browse_scope
        records = screen._conversations_state.page_records
        generation = screen._library_navigation_context_generation
        entered = asyncio.Event()

        async def veto():
            entered.set()
            return False

        screen._flush_active_file_notes = veto
        if incoming == "invalid_repair":
            rejected = {constants.LIBRARY_NAV_CONTEXT_CHARACTER_REPAIR: {}}
        elif incoming == "invalid_character":
            rejected = {constants.LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION: {}}
        elif incoming == "ordinary_veto":
            rejected = {LIBRARY_NAV_CONTEXT_MODE: "search"}
        else:
            rejected = {
                constants.LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION: character_navigation.serialize_library_unavailable_inspection(
                    character_navigation.LibraryUnavailableConversationInspection(
                        UnresolvedConversationKey(authority, "replacement-chat"),
                        route.return_target,
                    )
                )
            }
        if incoming == "superseded_typed":
            release = asyncio.Event()

            async def delayed_save():
                entered.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    await asyncio.wait_for(release.wait(), 2)
                return True

            async def permit():
                return True

            screen._flush_active_file_notes = delayed_save
            screen.apply_navigation_context(rejected)
            await asyncio.wait_for(entered.wait(), 2)
            attempted_generation = screen._library_navigation_context_generation
            assert screen._navigation_controller.character_route is committed
            assert not screen._conversations_state.loading
            screen._flush_active_file_notes = permit
            screen.apply_navigation_context(context)
            for _ in range(40):
                await pilot.pause(0.05)
                if (
                    screen._navigation_controller.character_route is not committed
                    and screen._pending_library_character_navigation is None
                ):
                    break
            replacement = screen._navigation_controller.character_route
            assert replacement is not None and replacement is not committed
            successful_generation = screen._library_navigation_context_generation
            assert successful_generation > attempted_generation > generation
            release.set()
            await pilot.pause(0.2)
            assert screen._navigation_controller.character_route is replacement
            assert (
                screen._library_navigation_context_generation == successful_generation
            )
            screen._unavailable_navigation.return_from_character(screen, committed)
            await pilot.pause()
            assert not returned
            committed = replacement
            scope = screen._library_unavailable_browse_scope
            records = screen._conversations_state.page_records
        else:
            screen.apply_navigation_context(rejected)
        if incoming.endswith("veto"):
            await asyncio.wait_for(entered.wait(), 2)
        await pilot.pause()
        assert screen._navigation_controller.character_route is committed
        assert screen._library_navigation_context_generation >= generation
        assert screen._library_unavailable_browse_scope is scope
        assert screen._conversations_state.page_records == records
        assert (
            screen._unavailable_navigation._library_unavailable_browse_scope_is_current(
                screen, scope
            )
        )
        filter_input = screen.query_one("#library-conversations-filter", Input)
        filter_input.value = "Retained"
        filter_input.focus()
        await pilot.press("enter")
        for _ in range(40):
            await pilot.pause(0.05)
            if not screen._conversations_state.loading:
                break
        assert screen._conversations_state.projection == "unavailable_character"
        assert (
            screen._conversations_state.page_records[0]["conversation_id"]
            == "retained-chat"
        )
        back = screen.query_one("#library-character-back-console", Button)
        back.press()
        await pilot.pause()
        assert returned and all(message is returned[0] for message in returned)
        assert (returned[0].screen_name, returned[0].screen_context) == (
            "chat",
            {"return_focus": "console-context-character"},
        )


@pytest.mark.asyncio
async def test_character_authority_read_yields_ui_and_cannot_admit_superseded_route():
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    entered, release = threading.Event(), threading.Event()
    read_threads = []

    def delayed_authority():
        read_threads.append(threading.get_ident())
        entered.set()
        release.wait(0.4)
        return "slow-authority"

    app.chachanotes_db = SimpleNamespace(get_local_authority_id=delayed_authority)
    payload = character_navigation.serialize_library_unavailable_inspection(
        character_navigation.LibraryUnavailableConversationInspection(
            UnresolvedConversationKey("slow-authority", "stale-chat"),
            character_navigation.RoleplayReturnTarget.console_context_character(),
        )
    )
    host = DestinationHarness(app, "library")
    try:
        async with host.run_test(size=(120, 40)) as pilot:
            screen = _active_destination_screen(host)
            await _wait_for_library_shell_ready(screen, pilot)
            started = time.monotonic()
            screen.apply_navigation_context(
                {constants.LIBRARY_NAV_CONTEXT_CHARACTER_INSPECTION: payload}
            )
            assert time.monotonic() - started < 0.1
            assert await asyncio.to_thread(entered.wait, 1)
            heartbeat = asyncio.Event()
            asyncio.get_running_loop().call_soon(heartbeat.set)
            await asyncio.wait_for(heartbeat.wait(), 0.1)
            assert screen._conversations_state.loading
            screen.apply_navigation_context({LIBRARY_NAV_CONTEXT_MODE: "search"})
            release.set()
            await pilot.pause(0.2)
            assert screen._selected_conversation_id != "stale-chat"
            assert read_threads and threading.get_ident() not in read_threads
    finally:
        release.set()


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
        assert (
            str(screen.query_one("#library-header-line").renderable)
            == "Library | Local"
        )
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
