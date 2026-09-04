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

from tldw_chatbook import config as app_config
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
from Tests.UI.app_factory import _build_test_app


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

        results_calls = _spy(
            monkeypatch, screen, "_refresh_library_rag_results_widgets"
        )
        history_calls = _spy(monkeypatch, screen, "_refresh_library_rag_history_widget")
        status_calls = _spy(
            monkeypatch, screen, "_refresh_library_rag_query_status_widgets"
        )

        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = "policy question"
        await screen.update_library_rag_query(
            Input.Changed(query_input, query_input.value)
        )

        assert results_calls == []
        assert history_calls == []
        assert status_calls == [True]

        # The run gate itself did update: a valid, non-empty query with an
        # available scope enables Run.
        await _wait_for_library_rag_query_ready(screen, pilot, "policy question")


@pytest.mark.asyncio
async def test_query_edit_leaves_landed_results_visible_and_in_sync(monkeypatch):
    """Typing a new (unsubmitted) query must not desync the visible Evidence
    rows from ``_rag_search_state.results`` -- the widget is deliberately
    left alone, so the backing list must be too (otherwise a click on an
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
        await screen.update_library_rag_query(
            Input.Changed(query_input, query_input.value)
        )
        await _wait_for_library_rag_query_ready(screen, pilot, "first query")

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        landed_results = screen._rag_search_state.results
        assert len(landed_results) == 1
        result_widget_before = screen.query_one("#library-rag-result-0")

        results_calls = _spy(
            monkeypatch, screen, "_refresh_library_rag_results_widgets"
        )
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
        assert screen._rag_search_state.results == landed_results
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
        {
            "results": [
                {"document_title": "Result A", "snippet": "s", "source_id": "id-1"}
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
        await screen.update_library_rag_query(
            Input.Changed(query_input, query_input.value)
        )
        await _wait_for_library_rag_query_ready(screen, pilot, "first query")

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")
        assert screen._rag_search_state.retrieval_status == "ready"

        query_input.value = "second query"
        await screen.update_library_rag_query(
            Input.Changed(query_input, query_input.value)
        )
        await pilot.pause()

        run_button = screen.query_one("#library-rag-run-query", Button)
        assert run_button.disabled is False
        assert str(run_button.label) != "Searching…"
        # Results are still the OLD landed set (B5 contract) -- only the
        # in-flight status was cleared, not the results themselves.
        assert len(screen._rag_search_state.results) == 1


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
        await screen.update_library_rag_query(
            Input.Changed(query_input, query_input.value)
        )
        await _wait_for_library_rag_query_ready(screen, pilot, "policy question")

        results_calls = _spy(
            monkeypatch, screen, "_refresh_library_rag_results_widgets"
        )
        history_calls = _spy(monkeypatch, screen, "_refresh_library_rag_history_widget")

        await screen.submit_library_rag_query(
            Input.Submitted(query_input, query_input.value)
        )
        await _wait_for_selector(screen, pilot, "#library-rag-result-0")

        assert results_calls  # the full refresh path still rebuilds results...
        assert history_calls  # ...and history.
        assert len(screen._rag_search_state.results) == 1


def test_refresh_search_rag_panel_state_widgets_skips_results_and_history_when_asked():
    """Unit-level pin on the new parameter's default and gating (no pilot needed)."""
    import inspect

    signature = inspect.signature(LibraryScreen._refresh_search_rag_panel_state_widgets)
    assert signature.parameters["include_results_and_history"].default is True


@pytest.mark.asyncio
async def test_rag_mode_query_edit_never_remounts_the_landed_answer(monkeypatch):
    """AC#1, extended to PR-3's Answer region (Task 4 review).

    The answer region is rebuilt on EVERY panel refresh, including the cheap
    per-keystroke one -- it has to be, or the in-flight "Generating answer…"
    line would outlive the flag the keystroke path clears. But an answer only
    ever changes when generation settles, so re-mounting it per character is
    the same pure waste task-284 removed for results/history, just with up to
    8,000 characters of `Static` instead of 100+ rows. The keyboard pilots
    above never caught it because they all run in keyword mode, where the
    region does not exist at all.

    The ONLY `rag`-mode test in this file, so it needs its own resolvable
    credential (PR-T2 Task 7 made `library_rag_answer_provider_ready` also
    check `Chat/provider_readiness.get_provider_readiness`, not just an
    endpoint NAME) rather than a file-wide autouse fixture -- see
    `test_product_maturity_gate16_library_search_rag.py`'s `_ready_library_
    rag_provider` for the same pattern applied at file scope there.
    """
    from Tests.UI.test_product_maturity_gate16_library_search_rag import (
        RecordingAnswerChat,
        StaticLibraryRagSearchService,
        _rag_result_fixture,
        _switch_to_rag_mode,
    )

    monkeypatch.setattr(app_config, "default_api_endpoint", "openai", raising=False)
    real_load_settings = app_config.load_settings

    def _load_settings_with_ready_openai_key(*args, **kwargs):
        settings = dict(real_load_settings(*args, **kwargs))
        api_settings = dict(settings.get("api_settings") or {})
        openai_settings = dict(api_settings.get("openai") or {})
        openai_settings["api_key"] = "sk-test-keystroke-ready-key"
        api_settings["openai"] = openai_settings
        settings["api_settings"] = api_settings
        return settings

    monkeypatch.setattr(
        app_config, "load_settings", _load_settings_with_ready_openai_key
    )

    app = _build_test_app()
    _seed_conversations(app, _two_conversations())
    app.library_rag_search_service = StaticLibraryRagSearchService(
        _rag_result_fixture()
    )
    app.library_rag_answer_chat = RecordingAnswerChat()
    host = LibraryHarness(app)

    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)

        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        await _switch_to_rag_mode(screen, pilot)

        query_input = screen.query_one("#library-rag-query-input", Input)
        query_input.value = "first query"
        await screen.update_library_rag_query(
            Input.Changed(query_input, query_input.value)
        )
        await _wait_for_library_rag_query_ready(screen, pilot, "first query")

        screen.query_one("#library-rag-run-query", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-answer-text")

        answer_region_before = screen.query_one("#library-rag-answer")
        answer_text_before = screen.query_one("#library-rag-answer-text")

        for suffix in (" refined", " again"):
            query_input.value += suffix
            await screen.update_library_rag_query(
                Input.Changed(query_input, query_input.value)
            )
            await pilot.pause()

        # Same widget instances -- proves no remove()/mount() cycle happened.
        assert screen.query_one("#library-rag-answer") is answer_region_before
        assert screen.query_one("#library-rag-answer-text") is answer_text_before
        assert screen._rag_search_state.answer is not None
