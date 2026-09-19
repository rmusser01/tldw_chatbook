"""Keep the paid disclosure and Run gate coherent across snapshot refreshes."""

from __future__ import annotations

import asyncio

import pytest
from textual.widgets import Button, Static

from Tests.UI.app_factory import _build_test_app
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.Library.library_rag_state import LibraryRagPanelState
from tldw_chatbook.Widgets.Library.library_search_rag_panel import (
    library_rag_query_quiet_text,
)


def _state(
    *, sources: int = 1, provider: str = "openai", mode: str = "rag"
) -> LibraryRagPanelState:
    return LibraryRagPanelState.from_values(
        source_counts={"notes": sources},
        query="What changed?",
        mode=mode,
        provider_name=provider,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("stale", "latest"),
    [
        (_state(sources=0), _state()),
        (_state(), _state(sources=0)),
        (_state(), _state(provider="anthropic")),
    ],
    ids=["sources-arrive", "sources-disappear", "provider-changes"],
)
async def test_snapshot_gate_survives_a_suspended_status_refresh(
    monkeypatch, stale, latest
):
    """A real conditional removal supplies the yield point, without timing sleeps."""
    host = LibraryProductionCSSHarness(_build_test_app())
    current = [_state(provider="")]
    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        monkeypatch.setattr(screen, "_library_rag_panel_state", lambda: current[0])
        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-blocked-callout")
        quiet = screen.query_one("#library-rag-query-quiet-line", Static)
        callout = screen.query_one("#library-rag-query-blocked-callout", Static)
        original_remove = callout.remove
        removed, release = asyncio.Event(), asyncio.Event()

        async def held_remove():
            await original_remove()
            removed.set()
            await release.wait()

        monkeypatch.setattr(callout, "remove", held_remove)
        current[0] = stale
        refresh = asyncio.create_task(
            screen._refresh_search_rag_panel_state_widgets(
                include_results_and_history=False
            )
        )
        try:
            await asyncio.wait_for(removed.wait(), timeout=5)
            current[0] = latest
            screen._sync_library_rag_scope_toggle_and_run_gate_widgets()
            during = (
                screen.query_one("#library-rag-run-query", Button).disabled,
                tuple(
                    str(row.render())
                    for row in screen.query("#library-rag-query-quiet-line")
                ),
            )
        finally:
            release.set()
            await asyncio.wait_for(refresh, timeout=5)

        expected = (
            not latest.query_state.run_action.enabled,
            (library_rag_query_quiet_text(latest),),
        )
        assert during == expected
        assert (
            screen.query_one("#library-rag-run-query", Button).disabled == expected[0]
        )
        assert (
            str(screen.query_one("#library-rag-query-quiet-line", Static).render())
            == expected[1][0]
        )
        assert screen.query_one("#library-rag-query-quiet-line", Static) is quiet


@pytest.mark.asyncio
async def test_snapshot_cannot_enable_run_without_its_disclosure_widget(monkeypatch):
    host = LibraryProductionCSSHarness(_build_test_app())
    async with host.run_test(size=(170, 50)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        monkeypatch.setattr(screen, "_library_rag_panel_state", _state)
        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-quiet-line")
        await screen.query_one("#library-rag-query-quiet-line", Static).remove()

        screen._sync_library_rag_scope_toggle_and_run_gate_widgets()

        assert screen.query_one("#library-rag-run-query", Button).disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(170, 50), (80, 24)])
@pytest.mark.parametrize("theme", ["textual-dark", "textual-light"])
@pytest.mark.parametrize("provider", ["openai", "local_transformers"])
@pytest.mark.parametrize("mode", ["rag", "search"])
async def test_ready_disclosure_is_fully_painted_without_moving_run(
    monkeypatch, size, theme, provider, mode
):
    host = LibraryProductionCSSHarness(_build_test_app())
    host.theme = theme
    current = [_state(provider=provider, mode=mode)]
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        monkeypatch.setattr(screen, "_library_rag_panel_state", lambda: current[0])
        screen.query_one("#library-row-browse-search", Button).press()
        await _wait_for_selector(screen, pilot, "#library-rag-query-quiet-line")
        quiet = screen.query_one("#library-rag-query-quiet-line", Static)
        region = quiet.region
        strips = list(screen._compositor.render_strips())
        painted = "\n".join(
            strips[y].crop(region.x, region.right).text
            for y in range(max(0, region.y), min(region.bottom, len(strips)))
        )
        assert painted.strip() == library_rag_query_quiet_text(current[0])
        run = screen.query_one("#library-rag-run-query", Button)
        assert not run.disabled
        run_y = run.region.y

        current[0] = _state(sources=0, provider=provider, mode=mode)
        screen._sync_library_rag_scope_toggle_and_run_gate_widgets()
        await pilot.pause()

        assert run.disabled
        assert run.region.y == run_y
        assert quiet.region.height == 1
