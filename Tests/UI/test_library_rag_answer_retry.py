"""Answer failure recovery stays visible beside the retained query."""

import asyncio
import threading

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _StaticLibraryRagSearchService,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    _rag_result_fixture,
    _ready_library_rag_provider,  # noqa: F401 - register the shared autouse fixture
    _switch_to_rag_mode,
    _wait_until,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("failure", ["exception", "empty"])
async def test_answer_failure_is_visible_beside_query_and_can_retry(
    size, theme, failure
):
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-42", "title": "Incident"}])
    app.library_rag_search_service = _StaticLibraryRagSearchService(
        _rag_result_fixture()
    )

    calls = []
    release = threading.Event()

    def answer(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            if failure == "exception":
                raise RuntimeError("Controlled generation failure")
            return ""
        assert release.wait(10), "Retry was never released"
        return "An expired credential caused the incident [S1]."

    app.library_rag_answer_chat = answer
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _switch_to_rag_mode(screen, pilot)
        field = screen.query_one("#library-rag-query-input", Input)
        field.value = "Why did the incident happen?"
        await _wait_for_library_rag_query_ready(screen, pilot, field.value)
        field.focus()
        await pilot.pause()
        scope = screen._library_rag_panel_state().scope.selected_source_types
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-rag-answer-error")
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen._rag_search_state.answer.status == "failed"
        assert screen.focused is field
        _assert_painted(screen, field)
        run = screen.query_one("#library-rag-run-query", Button)
        notice = screen.query_one("#library-rag-retrieval-notice", Static)
        disclosure = screen.query_one("#library-rag-query-quiet-line", Static)
        assert not run.disabled
        for widget in (notice, disclosure, run):
            _assert_painted(screen, widget)
        painted = " ".join(strip.text for strip in screen._compositor.render_strips())
        assert "Answer failed. Run again to retry." in painted
        assert "To openai: question + evidence" in painted
        assert screen.query("#library-rag-result-card-0")
        error = str(screen.query_one("#library-rag-answer-error").render())
        assert (
            "Controlled generation failure" in error
            if failure == "exception"
            else "The model returned an empty answer." in error
        )

        try:
            await pilot.press("enter")
            await _wait_until(
                pilot, lambda: len(calls) == 2, "Retry never reached provider"
            )
            await pilot.wait_for_scheduled_animations()
            assert screen._rag_search_state.answer_in_flight
            assert run.disabled and not notice.display
            assert "Answer failed" not in str(notice.render())
        finally:
            release.set()
        await _wait_for_selector(screen, pilot, "#library-rag-answer-citation-note")
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen._rag_search_state.answer.citation_status == "validated"
        assert screen._rag_search_state.answer.status == "ready"
        assert "An expired credential" in str(
            screen.query_one("#library-rag-answer-text").render()
        )
        assert field.value == "Why did the incident happen?" and screen.focused is field
        assert screen._library_rag_panel_state().scope.selected_source_types == scope
        _assert_painted(screen, field)
        assert not run.disabled and not notice.display
        assert not screen.query("#library-rag-answer-error")
        assert len(app.library_rag_search_service.calls) == 2
        assert len(calls) == 2
