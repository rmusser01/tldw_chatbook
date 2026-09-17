"""Keyboard replay and clearing of producer-created Recent searches."""

import asyncio

import pytest
from textual.widgets import Button, Collapsible, Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _StaticLibraryRagSearchService,
    _two_media_items,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    _rag_result_fixture,
    _ready_library_rag_provider,  # noqa: F401 - shared autouse fixture
)


def _host(theme):
    app = _build_test_app()
    _seed_conversations(
        app,
        [],
        notes=[{"id": "note-42", "title": "Incident"}],
        media=_two_media_items(),
    )
    service = _StaticLibraryRagSearchService(_rag_result_fixture())
    app.library_rag_search_service = service
    answers = []

    def answer(**kwargs):
        answers.append(kwargs)
        return "The incident was caused by an expired credential. [S1]"

    app.library_rag_answer_chat = answer
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    return host, app, service, answers


async def _settle(screen, pilot):
    await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
    await pilot.wait_for_scheduled_animations()
    await pilot.pause()


async def _submit(screen, pilot, text):
    field = screen.query_one("#library-rag-query-input", Input)
    field.value = text
    await _wait_for_library_rag_query_ready(screen, pilot, text)
    field.focus()
    await pilot.pause()
    await pilot.press("enter")
    await _wait_for_selector(screen, pilot, "#library-rag-result-card-0")
    await _settle(screen, pilot)


async def _tab_to(screen, pilot, predicate):
    stops = []
    for _ in range(30):
        if predicate(screen.focused):
            _assert_painted(screen, screen.focused)
            return screen.focused
        await pilot.press("tab")
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        stops.append(getattr(screen.focused, "id", None))
    raise AssertionError(stops)


async def _open_history(screen, pilot):
    title = await _tab_to(
        screen,
        pilot,
        lambda focused: (
            type(focused).__name__ == "CollapsibleTitle"
            and getattr(focused.parent, "id", None) == "library-rag-history"
        ),
    )
    history = screen.query_one("#library-rag-history", Collapsible)
    if history.collapsed:
        await pilot.press("enter")
        await _settle(screen, pilot)
    assert not history.collapsed
    return title


async def _open(screen, pilot):
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-search").press()
    await _wait_for_selector(screen, pilot, "#library-rag-query-input")


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("mode", ["search", "rag"])
async def test_keyboard_replay_uses_current_mode_scope_and_visible_focus(
    theme, size, mode
):
    host, app, service, answers = _host(theme)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _submit(screen, pilot, "alpha [/archive]")
        await _submit(screen, pilot, "beta")
        assert screen._rag_search_state.history == ("beta", "alpha [/archive]")
        if mode == "rag":
            screen.query_one("#library-rag-mode-toggle", Button).focus()
            await pilot.press("enter")
            await _settle(screen, pilot)
        screen.query_one("#library-rag-scope-toggle-notes", Button).focus()
        await pilot.press("enter")
        await _settle(screen, pilot)
        scope = screen._library_rag_panel_state().scope.selected_source_types
        assert scope == ("media",)
        await _open_history(screen, pilot)
        row = await _tab_to(
            screen,
            pilot,
            lambda focused: getattr(focused, "id", None) == "library-rag-history-1",
        )
        assert str(row.label) == "alpha [/archive]"
        await pilot.press("enter")
        await _settle(screen, pilot)
        assert len(service.calls) == 3
        assert service.calls[-1]["query"] == "alpha [/archive]"
        assert service.calls[-1]["mode"] == mode
        assert tuple(service.calls[-1]["scope"]) == scope
        assert len(answers) == (1 if mode == "rag" else 0)
        assert screen._rag_search_state.history == ("alpha [/archive]", "beta")
        assert (
            screen.query_one("#library-rag-query-input", Input).value
            == "alpha [/archive]"
        )
        assert (
            screen.query_one("#library-search-input", Input).value == "alpha [/archive]"
        )
        panel = screen.query_one("#library-search-rag-panel")
        assert screen.focused is not None and panel in screen.focused.ancestors
        _assert_painted(screen, screen.focused)
        assert app.app_config["library"]["search"]["history"] == [
            "alpha [/archive]",
            "beta",
        ]


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("mode", ["search", "rag"])
async def test_keyboard_clear_retains_results_and_visible_history_focus(
    theme, size, mode
):
    host, app, service, answers = _host(theme)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        if mode == "rag":
            screen.query_one("#library-rag-mode-toggle", Button).focus()
            await pilot.press("enter")
            await _settle(screen, pilot)
        await _submit(screen, pilot, "alpha")
        results = screen._rag_search_state.results
        answer = screen._rag_search_state.answer
        if mode == "rag":
            assert answer is not None and answer.status == "ready"
        title = await _open_history(screen, pilot)
        await _tab_to(
            screen,
            pilot,
            lambda focused: getattr(focused, "id", None) == "library-rag-history-clear",
        )
        await pilot.press("enter")
        await _settle(screen, pilot)
        assert screen._rag_search_state.history == ()
        assert app.app_config["library"]["search"]["history"] == []
        assert screen._rag_search_state.results is results
        assert screen._rag_search_state.answer is answer
        assert screen.query_one("#library-rag-query-input", Input).value == "alpha"
        assert len(service.calls) == 1
        assert len(answers) == (1 if mode == "rag" else 0)
        assert screen.focused is title
        _assert_painted(screen, title)
        _assert_painted(screen, screen.query_one("#library-rag-history-empty"))


@pytest.mark.asyncio
@pytest.mark.parametrize("outside_panel", [False, True])
async def test_answer_reveal_respects_newer_focus(monkeypatch, outside_panel):
    host, _, service, answers = _host("textual-dark")
    async with host.run_test(size=(170, 48)) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _submit(screen, pilot, "alpha")
        screen.query_one("#library-rag-mode-toggle", Button).focus()
        await pilot.press("enter")
        await _settle(screen, pilot)
        await _open_history(screen, pilot)
        await _tab_to(
            screen,
            pilot,
            lambda focused: getattr(focused, "id", None) == "library-rag-history-0",
        )
        panel = screen.query_one("#library-search-rag-panel")
        pending = []
        after_refresh = screen.call_after_refresh

        def defer(callback, *args, **kwargs):
            if getattr(callback, "__name__", "") == "_reveal_focused_control":
                pending.append((callback, args, kwargs))
                return True
            return after_refresh(callback, *args, **kwargs)

        monkeypatch.setattr(screen, "call_after_refresh", defer)
        await pilot.press("enter")
        await _settle(screen, pilot)
        assert pending
        if outside_panel:
            await pilot.press("slash")
            target = screen.query_one("#library-search-input", Input)
        else:
            target = screen.query_one("#library-rag-query-input", Input)
            for _ in range(20):
                if screen.focused is target:
                    break
                await pilot.press("shift+tab")
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen.focused is target
        _assert_painted(screen, target)
        offset = panel.scroll_offset
        for callback, args, kwargs in pending:
            callback(*args, **kwargs)
        await pilot.pause()
        assert screen.focused is target and panel.scroll_offset == offset
        _assert_painted(screen, target)
        assert len(service.calls) == 2 and len(answers) == 1
