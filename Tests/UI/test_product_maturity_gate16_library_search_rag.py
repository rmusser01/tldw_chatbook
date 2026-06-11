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
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen

from Tests.UI.test_destination_shells import (
    DestinationHarness,
    StaticLibraryConversationScopeService,
    StaticLibraryMediaScopeService,
    StaticLibraryNotesScopeService,
    _active_destination_screen,
    _build_test_app,
    _visible_text,
    _wait_for_library_snapshot,
    _wait_for_selector,
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


async def _wait_for_inspector_selection(
    screen,
    pilot,
    title: str,
    *,
    timeout: float = 2.0,
) -> None:
    deadline = time.monotonic() + timeout
    expected = f"Selected: {title}"
    while time.monotonic() < deadline:
        selected_widgets = list(screen.query("#library-rag-selected-result"))
        console_buttons = list(screen.query("#library-rag-use-in-console"))
        if selected_widgets and console_buttons:
            selected_widget = selected_widgets[0]
            console_button = console_buttons[0]
            if (
                selected_widget.display is True
                and expected in str(selected_widget.renderable)
                and console_button.disabled is False
            ):
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
    app = _build_test_app()
    _seed_library_sources(app)
    seen_routes: list[str] = []
    host = DestinationHarness(app, "library", seen_routes=seen_routes)

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)
        source_browser = screen.query_one("#library-source-browser")
        source_detail = screen.query_one("#library-source-detail")
        source_inspector = screen.query_one("#library-source-inspector")

        assert len(screen.query("#library-search-rag-panel")) == 0

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        assert _active_destination_screen(host) is screen
        assert seen_routes == []
        assert screen.query_one("#library-source-browser") is source_browser
        assert screen.query_one("#library-source-detail") is source_detail
        assert screen.query_one("#library-source-inspector") is source_inspector
        assert screen.query_one("#library-rag-inspector")
        assert source_inspector.query_one("#library-rag-inspector")
        assert len(screen.query("#search-rag-container")) == 0

        for selector in (
            "#library-search-rag-panel",
            "#library-rag-source-scope",
            "#library-rag-query-input",
            "#library-rag-run-query",
            "#library-rag-results",
            "#library-rag-inspector",
            "#library-rag-use-in-console",
        ):
            assert screen.query_one(selector)

        active_mode_button = screen.query_one("#library-mode-search", Button)
        assert str(active_mode_button.label) == "Search/RAG"
        assert "Ask Library sources, inspect evidence, then send selected snippets to Console." in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_search_rag_panel_exposes_blocked_recovery_for_empty_query() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        query_input = screen.query_one("#library-rag-query-input", Input)
        run_button = screen.query_one("#library-rag-run-query", Button)
        console_button = screen.query_one("#library-rag-use-in-console", Button)
        visible_text = _visible_text(screen)

        assert query_input.value == ""
        assert str(run_button.label) == "Run Search/RAG"
        assert run_button.disabled is True
        assert "Enter a question or search query" in str(run_button.tooltip)
        assert console_button.disabled is True
        assert "Library | Search/RAG | Blocked: Enter a question or search query. | Local" in visible_text
        assert "Scope: all local | Notes 1 | Media 1 | Conversations 1" in visible_text
        assert "Query" in visible_text
        assert "Enter: run query | Tab: move panes | Enter on result: select | u: Use in Console" in visible_text
        assert "Enter a question or search query" in visible_text
        assert "Run a query and select usable evidence before sending to Console" in visible_text


@pytest.mark.asyncio
async def test_library_search_rag_task_loop_orders_query_before_scope_and_results() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        panel = screen.query_one("#library-search-rag-panel")
        child_ids = [child.id for child in panel.children]
        assert child_ids.index("library-rag-query-controls") < child_ids.index(
            "library-rag-source-scope"
        )
        assert child_ids.index("library-rag-source-scope") < child_ids.index(
            "library-rag-results"
        )
        assert "Scope: all local | Notes 1 | Media 1 | Conversations 1" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_search_rag_mode_hides_generic_hub_and_study_actions() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        assert len(screen.query("#library-content-hub-title")) == 1

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        visible_text = _visible_text(screen)
        assert len(screen.query("#library-content-hub-title")) == 0
        assert len(screen.query("#library-hub-study-card")) == 0
        assert len(screen.query("#library-use-in-console")) == 0
        assert len(screen.query("#library-open-study")) == 0
        open_search_button = screen.query_one("#library-open-search", Button)
        assert open_search_button.display is False
        assert "Library Content Hub" not in visible_text
        assert "Knowledge workflow" not in visible_text
        assert "Study Dashboard" not in visible_text
        assert "Retrieval Inspector" in visible_text

        screen.query_one("#library-mode-sources", Button).press()
        await _wait_for_selector(screen, pilot, "#library-content-hub-title")
        assert screen.query_one("#library-open-search", Button) is open_search_button
        assert open_search_button.display is True


@pytest.mark.asyncio
async def test_library_search_rag_empty_sources_has_mode_local_blocked_status() -> None:
    app = _build_test_app()
    app.notes_scope_service = StaticLibraryNotesScopeService([])
    app.media_reading_scope_service = StaticLibraryMediaScopeService([])
    app.chat_conversation_scope_service = StaticLibraryConversationScopeService([])
    host = DestinationHarness(app, "library")

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        visible_text = _visible_text(screen)
        assert "Library | Search/RAG | Blocked: no Library sources | Local" in visible_text
        assert "No sources." in visible_text
        assert "Next: Add or import Library sources before querying." in visible_text
        assert "Recovery checklist" in visible_text
        assert "1. Import Library sources." in visible_text
        assert "2. Run Search/RAG." in visible_text
        assert "3. Select evidence, then Use in Console." in visible_text
        assert "Why: Enter a question or search query." not in visible_text
        assert screen.query_one("#library-rag-open-import-export", Button)
        assert len(screen.query("#library-content-hub-title")) == 0

        screen.query_one("#library-rag-open-import-export", Button).press()
        await _wait_for_selector(screen, pilot, "#library-import-export-workflow-title")

        assert "Library | Import/Export | Empty | Local" in _visible_text(screen)


@pytest.mark.asyncio
async def test_library_search_rag_query_updates_action_and_survives_recompose() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")
    query = "What does the research note say?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False
        assert str(run_button.tooltip) == ""
        assert len(screen.query("#library-rag-query-recovery")) == 0

        screen.refresh(recompose=True)
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        await _wait_for_selector(screen, pilot, "#library-rag-inspector")

        assert len(screen.query("#library-search-rag-panel")) == 1
        assert len(screen.query("#library-rag-inspector")) == 1
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
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        assert service.calls == [
            {
                "query": query,
                "scope": ("notes", "media", "conversations"),
                "mode": "rag",
                "top_k": 5,
                "include_citations": True,
            }
        ]
        visible_text = _visible_text(screen)
        assert "Incident Review | score 0.930" in visible_text
        assert "Expired credential caused the incident." in visible_text
        assert "Incident Review p.2" in visible_text
        assert "Status: Ready" in visible_text
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
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        assert screen.query_one("#library-rag-use-in-console", Button).disabled is True

        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_inspector_selection(screen, pilot, "Incident Review")

        assert screen.query_one("#library-rag-use-in-console", Button).disabled is False
        assert "Selected: Incident Review" in _visible_text(screen)

        screen.query_one("#library-rag-use-in-console", Button).press()
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
async def test_library_search_rag_selected_result_inspector_exposes_evidence_metadata() -> None:
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
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_inspector_selection(screen, pilot, "Incident Review")

        for selector in (
            "#library-rag-selected-snippet",
            "#library-rag-selected-citations",
            "#library-rag-selected-source",
            "#library-rag-selected-authority",
            "#library-rag-selected-eligibility",
            "#library-rag-selected-handoff",
        ):
            assert screen.query_one(selector, Static).display is True

        visible_text = _visible_text(screen)
        assert "note | workspace-a | 1 citation | eligible" in visible_text
        assert screen.query_one("#library-rag-use-selected-in-console", Button).disabled is False
        assert "Snippet: Expired credential caused the incident." in visible_text
        assert "Citations: Incident Review p.2" in visible_text
        assert "Source: note-42 / chunk-7" in visible_text
        assert "Score: 0.930" in visible_text
        assert "Runtime: local-fts" in visible_text
        assert "Retrieval" in visible_text
        assert "Selected Evidence" in visible_text
        assert "Authority & eligibility" in visible_text
        assert "Authority: Workspace: workspace-a" in visible_text
        assert "Eligibility: available for active workspace" in visible_text
        assert "Console handoff" in visible_text
        assert "Handoff: snippet + citations + source/chunk IDs" in visible_text


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
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
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
        await _wait_for_inspector_selection(screen, pilot, "Keyboard Evidence")

        console_button = screen.query_one("#library-rag-use-in-console", Button)
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
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
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
        await _wait_for_inspector_selection(screen, pilot, "Shortcut Evidence")

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
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        screen.query_one("#library-rag-select-result-0", Button).press()
        await _wait_for_inspector_selection(screen, pilot, "Server Incident Review")

        screen.query_one("#library-rag-use-in-console", Button).press()
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
async def test_library_search_rag_inspector_pre_mounts_selection_states() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    app.library_rag_search_service = StaticLibraryRagSearchService(
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
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")

        empty_state = screen.query_one("#library-rag-inspector-empty", Static)
        selected_state = screen.query_one("#library-rag-selected-result", Static)
        assert empty_state.display is True
        assert selected_state.display is False

        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)
        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        screen.query_one("#library-rag-select-result-0", Button).press()

        await _wait_for_inspector_selection(screen, pilot, "Incident Review")

        assert screen.query_one("#library-rag-inspector-empty", Static) is empty_state
        assert screen.query_one("#library-rag-selected-result", Static) is selected_state
        assert empty_state.display is False
        assert selected_state.display is True
        assert "Selected: Incident Review" in str(selected_state.renderable)


@pytest.mark.asyncio
async def test_library_search_rag_run_query_renders_persistent_recovery_without_service() -> None:
    app = _build_test_app()
    _seed_library_sources(app)
    host = DestinationHarness(app, "library")
    query = "What policy applies?"

    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_destination_screen(host)
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-service-error")

        visible_text = _visible_text(screen)
        assert "Unavailable: Library Search/RAG retrieval." in visible_text
        assert "Owner: Library retrieval service." in visible_text
        assert "Status: Blocked" in visible_text


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
        await _wait_for_library_snapshot(screen, pilot)

        screen.query_one("#library-mode-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-search-rag-panel")
        screen.query_one("#library-rag-query-input", Input).value = query
        await _wait_for_query_ready(screen, pilot, query)
        panel = screen.query_one("#library-search-rag-panel")
        inspector = screen.query_one("#library-rag-inspector")

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-searching")

        assert screen.query_one("#library-search-rag-panel") is panel
        assert screen.query_one("#library-rag-inspector") is inspector

        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        assert screen.query_one("#library-search-rag-panel") is panel
        assert screen.query_one("#library-rag-inspector") is inspector


@pytest.mark.asyncio
async def test_library_search_rag_worker_completion_ignores_unmounted_screen(monkeypatch) -> None:
    app = _build_test_app()
    screen = LibraryScreen(app)
    screen._library_rag_query = "Find evidence"

    async def fail_sync():
        raise AssertionError("unmounted worker completion should not touch the DOM")

    monkeypatch.setattr(screen, "_sync_search_rag_panel", fail_sync)

    await screen._apply_library_rag_search_outcome(
        LibraryRagSearchRequest(
            query="Find evidence",
            source_types=("notes",),
        ),
        LibraryRagSearchOutcome(status="ready"),
    )
