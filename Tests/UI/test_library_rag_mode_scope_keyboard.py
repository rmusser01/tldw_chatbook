"""Keyboard mode/scope changes retain the visible initiating control."""

import asyncio
import threading

import pytest
from textual.widgets import Button, Input

from Tests.UI.test_library_rag_history_keyboard import _host, _open, _settle, _submit
from Tests.UI.test_library_rag_result_focus import _assert_painted
from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_rag_query_ready,
)
from Tests.UI.test_product_maturity_gate16_library_search_rag import (
    _rag_result_fixture,
    _ready_library_rag_provider,  # noqa: F401 - shared autouse fixture
)


@pytest.mark.asyncio
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("kind", ["mode", "scope"])
async def test_keyboard_toggle_and_reverse_keep_visible_focus(theme, size, kind):
    host, _, service, answers = _host(theme)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        await _submit(screen, pilot, "incident")
        panel = screen.query_one("#library-search-rag-panel")
        selector = (
            "#library-rag-mode-toggle"
            if kind == "mode"
            else "#library-rag-scope-toggle-notes"
        )
        button = screen.query_one(selector, Button)
        original_label = str(button.label)
        button.focus()
        await pilot.pause()
        for changed in (True, False):
            await pilot.press("enter")
            await _settle(screen, pilot)
            current = screen.query_one(selector, Button)
            assert screen.query_one("#library-search-rag-panel") is panel
            assert screen.focused is current
            _assert_painted(screen, current)
            assert (str(current.label) != original_label) is changed
            assert (
                screen.query_one("#library-rag-query-input", Input).value == "incident"
            )
            assert screen.query_one("#library-search-input", Input).value == "incident"
            assert screen._rag_search_state.history == ("incident",)
            assert len(service.calls) == 1 and answers == []
            state = screen._library_rag_panel_state()
            if kind == "mode":
                assert state.query_state.mode == ("rag" if changed else "search")
                assert not state.results
            else:
                assert ("notes" in state.scope.selected_source_types) is not changed
                assert bool(state.results) is not changed
        # Continue from the retained control through the normal keyboard order.
        await pilot.press("tab")
        await _settle(screen, pilot)
        assert screen.focused is not current
        assert panel in screen.focused.ancestors
        _assert_painted(screen, screen.focused)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mode", "scope"])
@pytest.mark.parametrize("outside_panel", [False, True])
async def test_deferred_toggle_restore_respects_newer_focus(
    monkeypatch, kind, outside_panel
):
    host, _, service, answers = _host("textual-dark")
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        panel = screen.query_one("#library-search-rag-panel")
        pending = []
        monkeypatch.setattr(panel, "queue_after_recompose", pending.append)
        selector = (
            "#library-rag-mode-toggle"
            if kind == "mode"
            else "#library-rag-scope-toggle-notes"
        )
        screen.query_one(selector, Button).focus()
        await pilot.press("enter")
        await _settle(screen, pilot)
        assert pending
        if outside_panel:
            await pilot.press("slash")
            target = screen.query_one("#library-search-input", Input)
        else:
            target = screen.query_one("#library-rag-query-input", Input)
            target.focus()
        await pilot.pause()
        assert screen.focused is target
        _assert_painted(screen, target)
        offset = panel.scroll_offset
        for callback in pending:
            callback()
        await pilot.pause()
        assert screen.focused is target
        assert panel.scroll_offset == offset
        _assert_painted(screen, target)
        assert service.calls == [] and answers == []


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["mode", "scope"])
@pytest.mark.parametrize("phase", ["retrieval", "answer"])
async def test_keyboard_toggle_during_work_keeps_existing_staleness_contract(
    kind, phase
):
    host, app, _, _ = _host("textual-dark")
    gate = threading.Event()
    started = threading.Event()
    searches, answers = [], []

    async def search(query, scope, mode, **kwargs):
        searches.append((query, scope, mode))
        if phase == "retrieval":
            started.set()
            assert await asyncio.to_thread(gate.wait, 10)
        return _rag_result_fixture()

    def answer(**kwargs):
        answers.append(kwargs)
        if phase == "answer":
            started.set()
            assert gate.wait(10)
        return "An expired credential caused the incident. [S1]"

    app.library_rag_search_service = type(
        "GatedSearch", (), {"search": staticmethod(search)}
    )()
    app.library_rag_answer_chat = answer
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _open(screen, pilot)
        screen.query_one("#library-rag-mode-toggle", Button).focus()
        await pilot.press("enter")
        await _settle(screen, pilot)
        field = screen.query_one("#library-rag-query-input", Input)
        field.value = "incident"
        await _wait_for_library_rag_query_ready(screen, pilot, "incident")
        field.focus()
        await pilot.press("enter")
        try:
            await _wait_for_condition(
                pilot, started.is_set, message="Work did not start"
            )
            selector = (
                "#library-rag-mode-toggle"
                if kind == "mode"
                else "#library-rag-scope-toggle-notes"
            )
            old = screen.query_one(selector, Button)
            old.focus()
            await pilot.press("enter")
            await _wait_for_condition(
                pilot,
                lambda: (
                    screen.focused is screen.query_one(selector, Button)
                    and screen.focused is not old
                ),
                message="Toggle did not retain focus after recompose",
            )
            _assert_painted(screen, screen.focused)
        finally:
            gate.set()
        await _settle(screen, pilot)
        assert screen.focused is screen.query_one(selector, Button)
        _assert_painted(screen, screen.focused)
        assert searches == [("incident", ("notes", "media"), "rag")]
        assert screen._rag_search_state.history == ("incident",)
        assert screen._rag_search_state.query == "incident"
        if kind == "mode":
            assert screen._rag_search_state.mode == "search"
            assert screen._rag_search_state.results == ()
            assert screen._rag_search_state.answer is None
            assert screen._rag_search_state.retrieval_status == ""
            assert len(answers) == (phase == "answer")
        else:
            assert screen._rag_search_state.mode == "rag"
            assert len(screen._rag_search_state.results) == 1
            assert not screen._library_rag_panel_state().results
            assert screen._rag_search_state.answer.status == "ready"
            assert len(answers) == 1
        assert not screen._rag_search_state.answer_in_flight
