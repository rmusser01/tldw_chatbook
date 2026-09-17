"""Result arrival must not scroll the live keyboard target out of view."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Input

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _GatedLibraryRagSearchService,
    _seed_conversations,
    _wait_for_library_rag_query_ready,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _assert_painted(screen, widget):
    """Require the whole control inside its compositor clip, not just the DOM."""
    geometry = screen._compositor.visible_widgets.get(widget)
    assert geometry is not None, (widget.id, widget.region)
    region, clip = geometry
    assert region.intersection(clip) == region, (widget.id, region, clip)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("has_results", [True, False])
@pytest.mark.parametrize("new_focus", [None, "library-rag-mode-toggle"])
async def test_result_arrival_keeps_current_keyboard_control_painted(
    size, theme, new_focus, has_results
):
    """Unconditional Evidence reveal hides the submitter or a newer focus choice."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-1", "title": "Tides"}])
    service = _GatedLibraryRagSearchService(
        {
            "results": [
                {
                    "document_title": "Tides",
                    "snippet": "tidal evidence",
                    "source_id": "note-1",
                }
            ]
            if has_results
            else []
        }
    )
    app.library_rag_search_service = service
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
        _assert_painted(screen, field)
        try:
            await pilot.press("enter")
            for _ in range(100):
                if service.calls:
                    break
                await pilot.pause(0.02)
            assert service.calls
            _assert_painted(screen, field)
            target = field
            if new_focus:
                target = screen.query_one(f"#{new_focus}")
                target.focus()
                await pilot.pause()
        finally:
            service.release_event.set()
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen.focused is target
        assert field.value == "tides"
        _assert_painted(screen, target)

        if not has_results:
            assert screen._rag_search_state.retrieval_status == "empty"
            return

        # Preserving the query must leave an actual keyboard route to Evidence.
        for _ in range(40):
            if getattr(screen.focused, "id", None) == "library-rag-result-card-0":
                break
            await pilot.press("tab")
        else:
            pytest.fail("Tab never reached the evidence card")
        await pilot.pause()
        card = screen.focused
        assert card in screen._compositor.visible_widgets
        assert "Tides" in " ".join(
            strip.text for strip in screen._compositor.render_strips()
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 48), (80, 24)])
@pytest.mark.parametrize("foreign_focus", [False, True])
async def test_result_arrival_reveals_evidence_without_focused_panel_control(
    size, foreign_focus
):
    """Skipping all reveals would strand results below the fold in this path."""
    app = _build_test_app()
    _seed_conversations(app, [], notes=[{"id": "note-1", "title": "Tides"}])
    service = _GatedLibraryRagSearchService(
        {"results": [{"document_title": "Tides", "source_id": "note-1"}]}
    )
    app.library_rag_search_service = service
    host = LibraryProductionCSSHarness(app)
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
        try:
            await pilot.press("enter")
            for _ in range(100):
                if service.calls:
                    break
                await pilot.pause(0.02)
            assert service.calls
            target = (
                screen.query_one("#library-search-input") if foreign_focus else None
            )
            screen.set_focus(target)
            await pilot.pause()
        finally:
            service.release_event.set()
        await asyncio.wait_for(screen.workers.wait_for_complete(), timeout=10)
        await pilot.wait_for_scheduled_animations()
        await pilot.pause()
        assert screen.focused is target
        _assert_painted(screen, screen.query_one("#library-rag-results-heading"))
        if target is not None:
            _assert_painted(screen, target)
