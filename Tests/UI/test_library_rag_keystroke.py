"""Tests for task-284: Library Search/RAG panel per-keystroke refresh cost.

``Input.Changed`` on the query box used to call the full panel refresh
(``_refresh_search_rag_panel_state_widgets``), tearing down and remounting
the Evidence results list + Recent-searches history (~100+ widgets) on
every keystroke, even though neither depends on unsubmitted query text
(search runs on ``Input.Submitted`` / the Run button). This file pins:

  * A keystroke spies-through to only the run-button/status refresh --
    the results/history rebuild functions are never called.
  * Existing, already-landed results stay visible (and internal state
    stays in sync with them) while typing a new, not-yet-submitted query.
  * ``Input.Submitted`` (and the Run button) still run the full
    submit flow, rebuilding results/history as before.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    LibraryHarness,
    _StaticLibraryRagSearchService,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_screen_navigation import _build_test_app


def _spy(monkeypatch, screen, method_name: str) -> list[bool]:
    """Wrap an async screen method so calls are recorded but still run."""
    calls: list[bool] = []
    original = getattr(screen, method_name)

    async def spy(*args, **kwargs):
        calls.append(True)
        return await original(*args, **kwargs)

    monkeypatch.setattr(screen, method_name, spy)
    return calls


@pytest.mark.asyncio
async def test_query_edit_never_touches_results_or_history_widgets(monkeypatch):
    """AC#1: Input.Changed updates only the run-button/status line."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        results_calls = _spy(monkeypatch, screen, "_refresh_library_rag_results_widgets")
        history_calls = _spy(monkeypatch, screen, "_refresh_library_rag_history_widget")
        status_calls = _spy(monkeypatch, screen, "_refresh_library_rag_query_status_widgets")

        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = "policy question"
        await screen.update_library_rag_query(Input.Changed(query_input, query_input.value))

        assert results_calls == []
        assert history_calls == []
        assert status_calls == [True]

        # The run gate itself did update: a valid, non-empty query with an
        # available scope enables Run.
        await _wait_for_library_rag_query_ready(screen, pilot, "policy question")


@pytest.mark.asyncio
async def test_query_edit_leaves_landed_results_visible_and_in_sync(monkeypatch):
    """Typing a new (unsubmitted) query must not desync the visible Evidence
    rows from ``_library_rag_results`` -- the widget is deliberately left
    alone, so the backing list must be too (otherwise a click on an
    already-visible row would silently no-op)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {
            "results": [
                {"document_title": "Result A", "snippet": "s", "source_id": "id-1"},
            ]
        }
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = "first query"
        await screen.update_library_rag_query(Input.Changed(query_input, query_input.value))
        await _wait_for_library_rag_query_ready(screen, pilot, "first query")

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        landed_results = screen._library_rag_results
        assert len(landed_results) == 1
        result_widget_before = screen.query_one("#library-rag-result-0")

        results_calls = _spy(monkeypatch, screen, "_refresh_library_rag_results_widgets")
        history_calls = _spy(monkeypatch, screen, "_refresh_library_rag_history_widget")

        # Type more text WITHOUT submitting -- results/history must not be
        # touched, and the backing state must stay exactly what's shown.
        query_input.value = "first query refined"
        await screen.update_library_rag_query(
            Input.Changed(query_input, query_input.value)
        )
        await pilot.pause()

        assert results_calls == []
        assert history_calls == []
        assert screen._library_rag_results == landed_results
        # Same widget instance -- proves no remove()/mount() cycle happened.
        assert screen.query_one("#library-rag-result-0") is result_widget_before


@pytest.mark.asyncio
async def test_query_edit_unsticks_run_gate_after_prior_search_settles(monkeypatch):
    """The narrower in-flight-status reset must still un-stick the run gate:
    typing after a completed search should not leave the Run button
    permanently disabled/"Searching..." even though results are preserved.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result A", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = "first query"
        await screen.update_library_rag_query(Input.Changed(query_input, query_input.value))
        await _wait_for_library_rag_query_ready(screen, pilot, "first query")

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        assert screen._library_rag_retrieval_status == "ready"

        query_input.value = "second query"
        await screen.update_library_rag_query(Input.Changed(query_input, query_input.value))
        await pilot.pause()

        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False
        assert str(run_button.label) != "Searching…"
        # Results are still the OLD landed set (B5 contract) -- only the
        # in-flight status was cleared, not the results themselves.
        assert len(screen._library_rag_results) == 1


@pytest.mark.asyncio
async def test_input_submitted_still_runs_the_full_refresh(monkeypatch):
    """AC#1 (contrast case): Submitted must still rebuild results/history."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    service = _StaticLibraryRagSearchService(
        {"results": [{"document_title": "Result", "snippet": "s", "source_id": "id-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")

        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = "policy question"
        await screen.update_library_rag_query(Input.Changed(query_input, query_input.value))
        await _wait_for_library_rag_query_ready(screen, pilot, "policy question")

        results_calls = _spy(monkeypatch, screen, "_refresh_library_rag_results_widgets")
        history_calls = _spy(monkeypatch, screen, "_refresh_library_rag_history_widget")

        await screen.submit_library_rag_query(Input.Submitted(query_input, query_input.value))
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        assert results_calls  # the full refresh path still rebuilds results...
        assert history_calls  # ...and history.
        assert len(screen._library_rag_results) == 1


def test_refresh_search_rag_panel_state_widgets_skips_results_and_history_when_asked():
    """Unit-level pin on the new parameter's default and gating (no pilot needed)."""
    import inspect

    signature = inspect.signature(LibraryScreen._refresh_search_rag_panel_state_widgets)
    assert signature.parameters["include_results_and_history"].default is True
