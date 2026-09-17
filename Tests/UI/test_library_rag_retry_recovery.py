"""Retrieval errors must explain retry beside the preserved query."""

from __future__ import annotations

import asyncio

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
from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState


class _FailedRetrieval:
    async def search(self, *_args, **_kwargs):
        raise RuntimeError("private backend failure detail")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("unavailable", [False, True])
async def test_retrieval_failure_is_visible_and_same_query_can_retry(
    size, theme, unavailable
):
    """A mounted off-screen error does not tell a keyboard user what happened."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-1", "title": "Tides"}])
    app.library_rag_search_service = None if unavailable else _FailedRetrieval()
    host = LibraryProductionCSSHarness(app)
    host.theme = theme
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        screen._rag_search_state.mode = "search"
        field = screen.query_one("#library-rag-query-input", Input)
        field.value = "tides"
        await _wait_for_library_rag_query_ready(screen, pilot, "tides")
        field.focus()
        await pilot.pause()
        scope = screen._library_rag_panel_state().scope.selected_source_types

        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-rag-service-error")
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen.focused is field and field.value == "tides"
        _assert_painted(screen, field)
        run = screen.query_one("#library-rag-run-query", Button)
        assert not run.disabled
        _assert_painted(screen, screen.query_one("#library-rag-retrieval-notice"))
        painted = " ".join(strip.text for strip in screen._compositor.render_strips())
        assert (
            "Retrieval unavailable" in painted
            if unavailable
            else "Retrieval failed" in painted
        )
        assert "private backend failure detail" not in painted

        service = _StaticLibraryRagSearchService(
            {"results": [{"document_title": "Tides", "source_id": "note-1"}]}
        )
        app.library_rag_search_service = service
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-rag-result-card-0")
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen.focused is field and field.value == "tides"
        assert screen._library_rag_panel_state().scope.selected_source_types == scope
        assert service.calls[0]["query"] == "tides"
        assert not run.disabled
        painted = " ".join(strip.text for strip in screen._compositor.render_strips())
        assert "Retrieval failed" not in painted
        assert "Retrieval unavailable" not in painted
        assert not screen.query("#library-rag-service-error")


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
async def test_retrieval_notice_keeps_paid_disclosure_visible(size, theme, monkeypatch):
    """A failure notice must not replace or clip the recipient shown before retry."""
    host = LibraryProductionCSSHarness(_build_test_app())
    host.theme = theme
    failed = LibraryRagPanelState.from_values(
        source_counts={"notes": 1},
        query="tides",
        mode="rag",
        provider_name="openai",
        retrieval_status="failed",
    )
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        monkeypatch.setattr(screen, "_library_rag_panel_state", lambda: failed)
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        field = screen.query_one("#library-rag-query-input")
        field.focus()
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        for selector in (
            "#library-rag-query-quiet-line",
            "#library-rag-retrieval-notice",
            "#library-rag-run-query",
        ):
            _assert_painted(screen, screen.query_one(selector))
        painted = " ".join(strip.text for strip in screen._compositor.render_strips())
        assert "To openai: question + evidence" in painted
        assert "Retrieval failed. Run again to retry." in painted


@pytest.mark.asyncio
async def test_retry_notice_is_not_restored_by_suspended_old_status_refresh(
    monkeypatch,
):
    """A late conditional-status rebuild must not resurrect the last failure."""
    host = LibraryProductionCSSHarness(_build_test_app())

    def state(provider, status):
        return LibraryRagPanelState.from_values(
            source_counts={"notes": 1},
            query="tides",
            mode="rag",
            provider_name=provider,
            retrieval_status=status,
        )

    current = [state("", "ready")]
    async with host.run_test(size=(80, 24)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        monkeypatch.setattr(screen, "_library_rag_panel_state", lambda: current[0])
        screen.query_one("#library-row-browse-search").press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-blocked-callout")
        callout = screen.query_one("#library-rag-query-blocked-callout")
        remove = callout.remove
        removed, release = asyncio.Event(), asyncio.Event()

        async def held_remove():
            await remove()
            removed.set()
            await release.wait()

        monkeypatch.setattr(callout, "remove", held_remove)
        current[0] = state("openai", "failed")
        refresh = asyncio.create_task(
            screen._refresh_search_rag_panel_state_widgets(
                include_results_and_history=False
            )
        )
        try:
            await asyncio.wait_for(removed.wait(), timeout=5)
            notice = screen.query_one("#library-rag-retrieval-notice", Static)
            assert notice.display
            current[0] = state("openai", "searching")
            screen._sync_library_rag_scope_toggle_and_run_gate_widgets()
            assert not notice.display
        finally:
            release.set()
            await asyncio.wait_for(refresh, timeout=5)
        assert not notice.display and not str(notice.render())
        assert screen.query_one("#library-rag-run-query", Button).disabled
