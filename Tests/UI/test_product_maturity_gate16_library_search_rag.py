"""Gate 1.6 Library-native Search/RAG mounted UI regressions."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.widgets import Button, Input, Static

from tldw_chatbook.Library.library_rag_state import LibraryRagResultRow
from tldw_chatbook.Library.library_rag_service import (
    LibraryRagSearchOutcome,
    LibraryRagSearchRequest,
)
from tldw_chatbook.Library.library_shell_state import LIBRARY_ROW_BROWSE_SEARCH
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

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
    """Wait for the Library rail shell (not the retired 3-pane workbench).

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


REPO_ROOT = Path(__file__).resolve().parents[2]
GATE16_EVIDENCE = Path(
    "Docs/superpowers/qa/product-maturity/phase-3/"
    "2026-05-07-gate-1-6-library-native-search-rag.md"
)
ROADMAP = Path("Docs/superpowers/trackers/product-maturity-roadmap.md")
PHASE_3_README = Path("Docs/superpowers/qa/product-maturity/phase-3/README.md")
TASK_10 = Path("backlog/tasks/task-10 - Product-Maturity-Phase-3-Knowledge-And-Study-Workflows.md")
TASK_10_8 = Path(
    "backlog/tasks/task-10.8 - "
    "Product-Maturity-Phase-3.8-Gate-1.6-Library-Native-Search-RAG.md"
)
TASK_10_8_5 = Path(
    "backlog/tasks/task-10.8.5 - Gate-1.6.5-Library-Search-RAG-QA-closeout.md"
)


def _repo_text(path: Path) -> str:
    return (REPO_ROOT / path).read_text(encoding="utf-8")


def _seed_library_sources(app) -> None:
    app.notes_scope_service = StaticLibraryNotesScopeService(
        [{"title": "Research Note", "id": "note-1"}]
    )
    app.media_reading_scope_service = StaticLibraryMediaScopeService(
        [{"title": "Transcript A", "id": "media-1"}]
    )
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService(
        [{"title": "Planning Chat", "id": "chat-1"}]
    )


class StaticLibraryRagSearchService:
    def __init__(self, result):
        self.result = result
        self.calls = []

    async def search(self, query, scope, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "scope": scope,
                "mode": mode,
                **kwargs,
            }
        )
        return self.result


class DelayedLibraryRagSearchService(StaticLibraryRagSearchService):
    async def search(self, query, scope, mode, **kwargs):
        self.calls.append(
            {
                "query": query,
                "scope": scope,
                "mode": mode,
                **kwargs,
            }
        )
        await asyncio.sleep(0.05)
        return self.result


async def _wait_for_query_ready(screen, pilot, query: str, *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        inputs = list(screen.query("#library-rag-query-input"))
        buttons = list(screen.query("#library-rag-run-query"))
        if inputs and buttons:
            input_widget = inputs[0]
            run_button = buttons[0]
            if input_widget.value == query and run_button.disabled is False:
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Library Search/RAG query readiness: {query!r}")


async def _wait_for_evidence_selected(
    screen,
    pilot,
    title: str,
    *,
    timeout: float = 2.0,
) -> None:
    """Wait for a result row to be selected in-panel.

    The retired 3-pane inspector column (``#library-rag-selected-result``,
    ``#library-rag-use-in-console``) is never mounted by the new canvas;
    ``LibrarySearchRagPanel`` now surfaces selection directly on the result
    row (``is-selected`` class) and its own per-result Console action
    (``#library-rag-use-selected-in-console``).
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        selected_rows = list(screen.query(".library-rag-result-row.is-selected"))
        console_buttons = list(screen.query("#library-rag-use-selected-in-console"))
        if selected_rows and console_buttons:
            if title in str(selected_rows[0].renderable) and console_buttons[0].disabled is False:
                await pilot.pause()
                return
        await pilot.pause(0.01)
    raise AssertionError(f"Timed out waiting for Library Search/RAG selection: {title!r}")


def test_library_search_rag_provenance_labels_escape_rich_markup() -> None:
    result = LibraryRagResultRow.from_result(
        {
            "document_title": "Markup Attempt",
            "snippet": "Adapter provenance should render literally.",
            "source_id": "note-markup",
            "chunk_id": "chunk-markup",
            "provenance": {
                "source_type": "[bold]spoof[/]",
                "workspace_ids": ("[red]workspace[/]",),
                "authority_label": "[green]trusted[/]",
                "eligibility_reason": "[blink]blocked[/]",
            },
        }
    )

    combined = " ".join(
        (
            result.row_badge_label,
            result.authority_display_label,
            result.eligibility_label,
        )
    )
    assert "[bold]spoof[/]" not in combined
    assert "[red]workspace[/]" not in combined
    assert "[green]trusted[/]" not in combined
    assert r"\[bold]spoof\[/]" in combined
    assert r"\[red]workspace\[/]" in combined
    assert r"\[green]trusted\[/]" in combined


@pytest.mark.asyncio
async def test_library_search_rag_mode_mounts_native_panel_without_leaving_library() -> None:
    """Selecting the Search rail row mounts ``LibrarySearchRagPanel`` inside
    the Library canvas without navigating away. The retired 3-pane workbench
    panes (``#library-source-browser/-detail/-inspector``) and the inspector
    column (``#library-rag-inspector``, ``#library-rag-use-in-console``) are
    never mounted by the new shell; the canvas host is ``#library-canvas``."""
    app = _build_test_app()
    _seed_library_sources(app)
    seen_routes: list[str] = []
    host = DestinationHarness(app, "library", seen_routes=seen_routes)

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        assert len(screen.query("#library-search-rag-panel")) == 0

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert _active_destination_screen(host) is screen
        assert seen_routes == []
        assert len(screen.query("#search-rag-container")) == 0

        canvas = screen.query_one("#library-canvas")
        for selector in (
            "#library-search-rag-panel",
            "#library-rag-source-scope",
            "#library-rag-query-input",
            "#library-rag-run-query",
            "#library-rag-results",
        ):
            assert canvas.query_one(selector)

        active_row_button = screen.query_one("#library-row-browse-search", Button)
        assert active_row_button.tooltip == "Search / RAG"
        assert active_row_button.has_class("library-rail-row-selected")


@pytest.mark.asyncio
async def test_library_search_rag_panel_exposes_blocked_recovery_for_empty_query() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        query_input = screen.query_one("#library-rag-query-input", Input)
        run_button = screen.query_one("#library-rag-run-query", Button)
        visible_text = _visible_text(screen)

        assert query_input.value == ""
        assert str(run_button.label) == "Run"
        assert run_button.disabled is True
        assert "Enter a question or search query" in str(run_button.tooltip)
        assert not screen.query(".library-rag-result-action")
        # A1: the empty-query gate is a single quiet line now -- no summary
        # Static, callout box, "Run disabled:" reason, or recovery dump.
        assert screen.query_one("#library-rag-query-quiet-line", Static)
        assert "Enter a question or search query." in visible_text
        assert not screen.query("#library-rag-query-blocked-callout")
        assert not screen.query("#library-rag-query-recovery")
        assert not screen.query("#library-rag-run-disabled-reason")
        assert "Blocked: enter a question or search query." not in visible_text
        assert "Blocked | Enter a question before running retrieval." not in visible_text
        assert "Scope: all local sources" in visible_text
        # A4: the retired-workbench shortcuts line is gone; Enter-to-run
        # keeps working (covered by the keyboard-enter pilot below).
        assert not screen.query("#library-rag-query-shortcuts")
        assert "Tab: move panes" not in visible_text


@pytest.mark.asyncio
async def test_library_search_rag_task_loop_orders_query_before_scope_and_results() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        panel = screen.query_one("#library-search-rag-panel")
        child_ids = [child.id for child in panel.children]
        assert child_ids.index("library-rag-query-controls") < child_ids.index(
            "library-rag-source-scope"
        )
        assert child_ids.index("library-rag-source-scope") < child_ids.index(
            "library-rag-results"
        )
        assert "Scope: all local sources" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_search_rag_empty_sources_has_mode_local_blocked_status() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        visible_text = _visible_text(screen)
        # (task-185) The no-sources state is ONE quiet gate line plus the
        # single Open Import media action -- the old 8-line recovery dump,
        # its checklist, the "Select at least one source." query line, and
        # the Evidence empty-state hints must not stack on top of it.
        gate_line = screen.query_one("#library-rag-scope-recovery", Static)
        assert (
            str(gate_line.renderable)
            == "No Library sources yet — import media or create notes, then search."
        )
        assert "No Library sources yet" in visible_text
        assert "Recovery checklist" not in visible_text
        assert "Owner: Library source index." not in visible_text
        assert "Unavailable: Library Search/RAG." not in visible_text
        assert "1. Import Library sources." not in visible_text
        assert "Select at least one source." not in visible_text
        assert "Why: Enter a question or search query." not in visible_text
        assert "No evidence yet. Run Search/RAG to populate results." not in visible_text
        assert (
            "Add or import sources, run a query, then select evidence for Console."
            not in visible_text
        )
        assert not screen.query("#library-rag-results-empty")
        assert not screen.query("#library-rag-evidence-empty-guidance")
        # The quiet-line slot stays mounted (empty) so the Run button's
        # position is stable, but it carries no second guidance layer.
        quiet_line = screen.query_one("#library-rag-query-quiet-line", Static)
        assert str(quiet_line.renderable) == ""
        recovery_button = screen.query_one("#library-rag-open-import-export", Button)
        assert str(recovery_button.label) == "Open Import media"
        assert recovery_button.tooltip == "Open Library Import media to add sources."
        # Pressing this button drives the shell selection to the Ingest ▸
        # Import media canvas row (the Import/Export row/mode it used to
        # target is retired); the canvas-switch behavior itself is covered
        # by test_library_shell.py.


@pytest.mark.asyncio
async def test_library_search_rag_query_updates_action_and_survives_recompose() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")
    query = "What does the research note say?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False
        assert str(run_button.tooltip) == ""
        assert len(screen.query("#library-rag-query-recovery")) == 0
        # (task-185) The gate helper's one-row slot stays mounted (empty)
        # once the query is valid, so the Run button never shifts when the
        # "Enter a question or search query." line clears.
        quiet_line = screen.query_one("#library-rag-query-quiet-line", Static)
        assert str(quiet_line.renderable) == ""

        screen.refresh(recompose=True)
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert len(screen.query("#library-search-rag-panel")) == 1
        assert screen.query_one("#library-rag-query-input", Input).value == query
        assert screen.query_one("#library-rag-run-query", Button).disabled is False


@pytest.mark.asyncio
async def test_library_search_rag_run_query_renders_service_results_and_calls_scope() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": "0.93",
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                    "provenance": {
                        "source_type": "note",
                        "workspace_ids": ("workspace-a",),
                        "active_workspace_id": "workspace-a",
                    },
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app.library_rag_search_service = service
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        # Default canvas mode is "search" (keyword); RAG-mode gating and
        # dispatch are covered separately by the mode-toggle pilots.
        assert service.calls == [
            {
                "query": query,
                "scope": ("notes", "media", "conversations"),
                "mode": "search",
                "top_k": 5,
                "include_citations": True,
            }
        ]
        visible_text = _visible_text(screen)
        assert "Incident Review | score 0.930" in visible_text
        assert "Expired credential caused the incident." in visible_text
        assert "Incident Review p.2" in visible_text
        assert len(screen.query("#library-rag-service-error")) == 0


@pytest.mark.asyncio
async def test_library_search_rag_selected_result_launches_console_live_work() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": 0.93,
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                    "provenance": {
                        "source_type": "note",
                        "workspace_ids": ("workspace-a",),
                        "active_workspace_id": "workspace-a",
                    },
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        assert not screen.query(".library-rag-console-action")

        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_evidence_selected(screen, pilot, "Incident Review")

        assert screen.query_one("#library-rag-use-selected-in-console", Button).disabled is False
        assert "Incident Review" in str(screen.query_one("#library-rag-result-0").renderable)

        screen.query_one("#library-rag-use-selected-in-console", Button).press()
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    launch_kwargs = app.open_console_for_live_work.call_args.kwargs
    assert launch_kwargs["source"] == "Library Search/RAG"
    assert launch_kwargs["title"] == "Incident Review"
    assert launch_kwargs["status"] == "staged"
    assert launch_kwargs["recovery"] == "Review citations before sending."
    assert launch_kwargs["action_label"] == "Review evidence in Console"
    payload = launch_kwargs["payload"]
    assert payload["target_id"] == "local:library-rag:note-42:chunk-7"
    assert payload["snippet"] == "Expired credential caused the incident."
    evidence_bundle = payload["evidence_bundle"]
    evidence_reference = evidence_bundle["references"][0]
    assert evidence_bundle["query"] == query
    assert evidence_bundle["status"] == "available"
    assert evidence_reference["evidence_id"] == "S1"
    assert evidence_reference["source_id"] == "note-42"
    assert evidence_reference["source_type"] == "note"
    assert evidence_reference["snippet"] == "Expired credential caused the incident."
    assert evidence_reference["authority_label"] == "Workspace: workspace-a"
    assert evidence_reference["metadata"]["active_context_eligible"] is True
    assert evidence_reference["metadata"]["global_browse_visible"] is True
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_library_search_rag_selected_result_evidence_metadata() -> None:
    """``LibrarySearchRagPanel`` surfaces the row provenance badge, snippet,
    and citations for a result unconditionally (not gated on selection). The
    retired inspector column's structured evidence breakdown (Source/Score/
    Runtime lines, "Authority & eligibility" and "Console Handoff" headings,
    the eligibility/handoff sentences) has no successor in the single-pane
    canvas."""
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "score": 0.93,
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                    "runtime_backend": "local-fts",
                    "citations": [{"label": "Incident Review p.2"}],
                    "provenance": {
                        "source_type": "note",
                        "workspace_ids": ("workspace-a",),
                        "active_workspace_id": "workspace-a",
                        "active_context_eligible": True,
                        "eligibility_reason": "active_workspace_match",
                    },
                }
            ],
            "runtime_backend": "local-fts",
        }
    )
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_evidence_selected(screen, pilot, "Incident Review")

        visible_text = _visible_text(screen)
        # UX wave M5: humanized badge composition -- "eligible" contributes
        # nothing (only a "blocked" -> "excluded from context" badge is
        # shown), joined with " · " instead of "|".
        assert "note · workspace-a · 1 citation" in visible_text
        assert screen.query_one("#library-rag-use-selected-in-console", Button).disabled is False
        assert "Expired credential caused the incident." in visible_text
        assert "Citations: Incident Review p.2" in visible_text


@pytest.mark.asyncio
async def test_library_search_rag_keyboard_enter_runs_query_and_handoff_button() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Keyboard Evidence",
                    "snippet": "Keyboard-only users can run and stage evidence.",
                    "source_id": "note-keyboard",
                    "chunk_id": "chunk-1",
                }
            ],
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "Can keyboard-only users stage evidence?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = query
        await _wait_for_query_ready(screen, pilot, query)

        query_input.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        select_button = screen.query_one("#library-rag-select-result-0", Button)
        select_button.focus()
        await pilot.press("enter")
        await _wait_for_evidence_selected(screen, pilot, "Keyboard Evidence")

        console_button = screen.query_one("#library-rag-use-selected-in-console", Button)
        console_button.focus()
        await pilot.press("enter")
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    payload = app.open_console_for_live_work.call_args.kwargs["payload"]
    assert payload["query"] == query
    assert payload["source_id"] == "note-keyboard"
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_library_search_rag_keyboard_u_shortcut_uses_selected_evidence() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Shortcut Evidence",
                    "snippet": "The u shortcut stages selected evidence.",
                    "source_id": "note-shortcut",
                    "chunk_id": "chunk-u",
                }
            ],
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "Can the u shortcut stage evidence?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = query
        await _wait_for_query_ready(screen, pilot, query)

        query_input.focus()
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        select_button = screen.query_one("#library-rag-select-result-0", Button)
        select_button.focus()
        await pilot.press("enter")
        await _wait_for_evidence_selected(screen, pilot, "Shortcut Evidence")

        await pilot.press("u")
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    payload = app.open_console_for_live_work.call_args.kwargs["payload"]
    assert payload["query"] == query
    assert payload["source_id"] == "note-shortcut"
    assert payload["chunk_id"] == "chunk-u"
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_library_search_rag_server_result_launches_server_console_live_work() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Server Incident Review",
                    "snippet": "Server retrieval found the authoritative incident record.",
                    "score": 0.88,
                    "source_id": "server-note-42",
                    "chunk_id": "chunk-9",
                    "runtime_backend": "server-rag",
                    "citations": [{"label": "Server Incident Review p.4"}],
                }
            ],
            "runtime_backend": "server-rag",
        }
    )
    app.open_console_for_live_work = Mock()
    app.open_chat_with_handoff = Mock()
    host = DestinationHarness(app, "library")
    query = "What did the server evidence say?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_evidence_selected(screen, pilot, "Server Incident Review")

        screen.query_one("#library-rag-use-selected-in-console", Button).press()
        await pilot.pause(0.1)

    app.open_console_for_live_work.assert_called_once()
    launch_kwargs = app.open_console_for_live_work.call_args.kwargs
    assert launch_kwargs["source"] == "Library Search/RAG"
    assert launch_kwargs["title"] == "Server Incident Review"
    assert launch_kwargs["status"] == "staged"
    assert launch_kwargs["recovery"] == "Review citations before sending."
    assert launch_kwargs["action_label"] == "Review evidence in Console"
    payload = launch_kwargs["payload"]
    assert payload["target_id"] == "server:library-rag:server-note-42:chunk-9"
    assert payload["result_id"] == "server-note-42:chunk-9"
    assert payload["query"] == query
    assert payload["title"] == "Server Incident Review"
    assert payload["source_id"] == "server-note-42"
    assert payload["chunk_id"] == "chunk-9"
    assert payload["snippet"] == "Server retrieval found the authoritative incident record."
    assert payload["citations"] == ["Server Incident Review p.4"]
    assert payload["score"] == 0.88
    assert payload["runtime_backend"] == "server-rag"
    assert payload["source_authority"] == "server"
    assert payload["source_selector_state"] == "server"
    evidence_reference = payload["evidence_bundle"]["references"][0]
    assert evidence_reference["source_owner"] == "server"
    assert evidence_reference["authority_label"] == "Source authority: server"
    assert evidence_reference["snippet"] == (
        "Server retrieval found the authoritative incident record."
    )
    app.open_chat_with_handoff.assert_not_called()


@pytest.mark.asyncio
async def test_library_search_rag_run_query_renders_persistent_recovery_without_service() -> None:
    """RAG mode with no ``_rag_service`` on the app is blocked at the
    query-readiness gate itself (``provider_ready``), never reaching the
    retrieval service -- the provider gate added in L3a front-runs the
    previous service-level "RAG unavailable" recovery for this scenario
    (the retrieval service still double-guards internally for non-UI
    callers, but the Run button is disabled before that code can run). The
    default canvas mode is "search" (keyword, always available given
    seeded local seams), so reaching this state requires explicitly
    cycling to RAG mode first.
    """
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")
    query = "What policy applies?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-mode-toggle")
        screen.query_one("#library-rag-mode-toggle", Button).press()

        # Mode toggle drives a full-screen recompose: poll for the new mode
        # label rather than assuming a fixed number of pauses is enough.
        for _ in range(150):
            toggles = list(screen.query("#library-rag-mode-toggle"))
            if toggles and str(toggles[0].label) == "mode: RAG Answer ▸":
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError("Mode toggle never switched to RAG Answer.")

        screen.query_one("#library-rag-query-input", Input).value = query
        # `.value` lands on the widget synchronously, but the screen's own
        # query state (and therefore the disabled reason) only catches up
        # once the Input.Changed handler runs -- poll for the settled
        # blocked reason rather than the transient "no query yet" one.
        for _ in range(150):
            if "provider/model before asking for a RAG answer" in _visible_text(screen):
                break
            await pilot.pause(0.02)
        else:
            raise AssertionError(
                f"RAG mode never blocked Run without a provider. "
                f"Visible text: {_visible_text(screen)}"
            )

        assert screen.query_one("#library-rag-run-query", Button).disabled is True
        assert not screen.query("#library-rag-service-error")


@pytest.mark.asyncio
async def test_library_search_rag_run_query_preserves_panel_instances_during_updates() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = DelayedLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Incident Review",
                    "snippet": "Expired credential caused the incident.",
                    "source_id": "note-42",
                    "chunk_id": "chunk-7",
                }
            ],
        }
    )
    host = DestinationHarness(app, "library")
    query = "Why did the incident happen?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_shell_ready(screen, pilot)

        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)
        panel = screen.query_one("#library-search-rag-panel")

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-searching-line")

        assert screen.query_one("#library-search-rag-panel") is panel

        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        assert screen.query_one("#library-search-rag-panel") is panel


@pytest.mark.asyncio
async def test_library_search_rag_worker_completion_ignores_unmounted_screen(monkeypatch) -> None:
    app = _build_test_app()
    screen = LibraryScreen(app)
    # Put the screen in the state where every guard in
    # _apply_library_rag_search_outcome passes EXCEPT is_mounted, so that the
    # mount guard is the sole thing preventing the DOM refresh. The code under
    # test refreshes via _refresh_search_rag_panel_state_widgets (not the stale
    # _sync_search_rag_panel), so that is the method the test must poison.
    screen._library_selected_row_id = LIBRARY_ROW_BROWSE_SEARCH
    screen._library_rag_query = "Find evidence"
    monkeypatch.setattr(screen, "query", lambda *args, **kwargs: [object()])

    async def fail_refresh() -> None:
        raise AssertionError("unmounted worker completion should not touch the DOM")

    monkeypatch.setattr(screen, "_refresh_search_rag_panel_state_widgets", fail_refresh)

    assert screen.is_mounted is False
    await screen._apply_library_rag_search_outcome(
        LibraryRagSearchRequest(
            query="Find evidence",
            source_types=("notes",),
        ),
        LibraryRagSearchOutcome(status="ready"),
    )
