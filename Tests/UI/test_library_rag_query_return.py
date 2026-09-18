"""Keyboard return to the current RAG query across resize transitions."""

import asyncio

import pytest
from textual.widgets import Input

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
    _ready_library_rag_provider,  # noqa: F401 - register shared autouse fixture
    _switch_to_rag_mode,
)

QUERY = "Why did the incident happen?"
REPLY = (
    "\n".join(
        f"Answer line {i:03d}: an expired credential caused the incident."
        for i in range(1, 81)
    )
    + " [S1]"
)


def _answer_host(theme):
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-42", "title": "Incident"}])
    service = _StaticLibraryRagSearchService(_rag_result_fixture())
    app.library_rag_search_service = service
    calls = []

    def answer(**kwargs):
        calls.append(kwargs)
        return REPLY

    app.library_rag_answer_chat = answer
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    return host, service, calls


async def _open_answer(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-search").press()
    await _switch_to_rag_mode(screen, pilot)
    field = screen.query_one("#library-rag-query-input", Input)
    field.value = QUERY
    await _wait_for_library_rag_query_ready(screen, pilot, QUERY)
    field.focus()
    await pilot.pause()
    await pilot.press("enter")
    await _wait_for_selector(screen, pilot, "#library-rag-answer-citation-note")
    await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
    await pilot.wait_for_scheduled_animations()
    await pilot.pause()
    return screen, field


async def _to_evidence_action(screen, pilot):
    for _ in range(15):
        await pilot.press("tab")
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        if getattr(screen.focused, "id", None) == "library-rag-result-card-0":
            break
    assert getattr(screen.focused, "id", None) == "library-rag-result-card-0"
    await pilot.press("pageup", "pageup", "pagedown", "pagedown", "tab")
    await pilot.wait_for_scheduled_animations()
    await pilot.pause()
    assert getattr(screen.focused, "id", None) == "library-rag-open-result-0"


async def _back_to_query(screen, pilot):
    for _ in range(15):
        if getattr(screen.focused, "id", None) == "library-rag-query-input":
            break
        await pilot.press("shift+tab")
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
    current = screen.query_one("#library-rag-query-input", Input)
    assert screen.focused is current
    _assert_painted(screen, current)
    return current


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("start", [(170, 48), (80, 24)])
@pytest.mark.parametrize("origin", ["query", "evidence"])
async def test_resize_retains_visible_focus_and_reverse_tab_returns_to_query(
    theme, start, origin
):
    host, service, calls = _answer_host(theme)
    async with host.run_test(size=start) as pilot:
        screen, field = await _open_answer(host, pilot)
        scope = screen._library_rag_panel_state().scope.selected_source_types
        state = screen._rag_search_state.answer
        if origin == "evidence":
            await _to_evidence_action(screen, pilot)
        retained = screen.focused
        _assert_painted(screen, retained)
        selection = field.selection
        sizes = (
            ((80, 24), (170, 48), (170, 24), (170, 48))
            if start == (170, 48)
            else ((170, 48), (80, 24), (80, 48), (80, 24))
        )
        observations = []
        for size in sizes:
            await pilot.resize_terminal(*size)
            await pilot.wait_for_scheduled_animations()
            await pilot.pause()
            observations.append(
                (size, getattr(screen.focused, "id", None), str(retained.region))
            )
            assert screen.focused is retained, observations
            _assert_painted(screen, retained)
            assert screen.query_one("#library-rag-query-input", Input) is field
            assert field.selection == selection
            assert screen._rag_search_state.answer is state
            assert state.text == REPLY
        assert await _back_to_query(screen, pilot) is field
        assert field.value == QUERY
        assert screen._rag_search_state.mode == "rag"
        assert screen._library_rag_panel_state().scope.selected_source_types == scope
        assert len(service.calls) == len(calls) == 1
        await pilot.press("end", "space", "x")
        await _wait_for_library_rag_query_ready(screen, pilot, QUERY + " x")
        assert field.value == QUERY + " x"
        assert screen.focused is field
        _assert_painted(screen, field)
        assert len(service.calls) == len(calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outside_panel", [False, True])
@pytest.mark.parametrize("size", [(80, 24), (170, 24)])
async def test_deferred_resize_respects_newer_focus(monkeypatch, outside_panel, size):
    host, service, calls = _answer_host("textual-dark")
    async with host.run_test(size=(170, 48)) as pilot:
        screen, field = await _open_answer(host, pilot)
        await _to_evidence_action(screen, pilot)
        panel = screen.query_one("#library-search-rag-panel")
        pending = []
        after_refresh = panel.call_after_refresh

        def defer(callback, *args, **kwargs):
            # Hold only resize reveals, not Textual's own focus-scroll work.
            if getattr(callback, "__name__", "") == "_reveal_focused_control":
                pending.append((callback, args, kwargs))
                return True
            return after_refresh(callback, *args, **kwargs)

        monkeypatch.setattr(panel, "call_after_refresh", defer)
        await pilot.resize_terminal(*size)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert pending
        if outside_panel:
            await pilot.press("slash")
            target = screen.query_one("#library-search-input", Input)
        else:
            target = await _back_to_query(screen, pilot)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen.focused is target
        _assert_painted(screen, target)
        offset = panel.scroll_offset
        for callback, args, kwargs in tuple(pending):
            callback(*args, **kwargs)
        await pilot.pause()
        assert screen.focused is target
        assert panel.scroll_offset == offset
        _assert_painted(screen, target)
        assert field.value == QUERY
        assert screen._rag_search_state.answer.text == REPLY
        assert len(service.calls) == len(calls) == 1
