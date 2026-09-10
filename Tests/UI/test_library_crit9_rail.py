"""Library rail fixes from critique #9 (tasks 32212, 32220, 32226, 32230, 32219).

Every geometry/paint assertion here runs against the PRODUCTION stylesheet
sequence (``LibraryProductionCSSHarness`` == ``TldwCli.CSS_PATH``), because the
rail's rules live in the ``screen_agentic_library.tcss`` split sheet -- a
bundle-only or widget-only harness sees none of them and would happily pass a
layout that is broken in the app.
"""

from __future__ import annotations

import pytest
from textual.widgets import Input, Static

from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _library_host() -> LibraryProductionCSSHarness:
    """A Library screen under the exact production stylesheet sequence."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    return LibraryProductionCSSHarness(app)


def _painted(host, region) -> str:
    """The painted text inside ``region``, one screen row per list entry."""
    strips = host.screen._compositor.render_strips()
    lines = ["".join(segment.text for segment in strip) for strip in strips]
    out = []
    for y in range(region.y, region.bottom):
        if 0 <= y < len(lines):
            out.append(lines[y][region.x : region.right])
    return "\n".join(out)


# --- task-32212: the rail search row must fit its pane ---------------------


#: The rail pane's own right border glyph (``border: solid $ds-column-line``).
_RAIL_FRAME_GLYPH = "│"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30), (60, 24)])
async def test_the_rail_search_row_never_pushes_the_canvas_frame(size) -> None:
    """AC#1/#2: the Input + clear button stay inside the rail's own width and
    every painted search-row line keeps the rail's frame column.

    The boxes were always laid out inside the rail (the regions below passed
    on dev); what did NOT fit was the clear button's own CONTENT -- see the
    painted-frame assertion, which is the one that reproduced the defect.
    """
    host = _library_host()
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        rail = screen.query_one("#library-rail")
        row = screen.query_one("#library-rail-search-row")
        box = screen.query_one("#library-search-input")
        clear = screen.query_one("#library-search-clear")
        assert row.region.right <= rail.region.right, (row.region, rail.region)
        assert clear.region.right <= rail.region.right, (clear.region, rail.region)
        assert box.region.right <= clear.region.x, (box.region, clear.region)
        canvas = screen.query_one("#library-canvas")
        if canvas.display:
            # Below the compact breakpoint the rail is single-stage and the
            # canvas is deliberately not mounted beside it (task-32066).
            assert canvas.region.x >= rail.region.right, (canvas.region, rail.region)

        # AC#2, the real pin: the painted frame. Every line the search row
        # covers must still carry the rail's right border in the rail's own
        # last column -- the regression painted the middle line two cells
        # long, shifting that border (and the canvas's left border) right.
        strips = host.screen._compositor.render_strips()
        lines = ["".join(segment.text for segment in strip) for strip in strips]
        frame_column = rail.region.right - 1
        for y in range(row.region.y, row.region.bottom):
            painted = lines[y]
            assert painted[frame_column] == _RAIL_FRAME_GLYPH, (
                f"search-row line y={y} lost the rail frame at column "
                f"{frame_column} at size {size}: {painted[:frame_column + 4]!r}"
            )


# --- task-32220: the rail heading is never cut mid-word --------------------


@pytest.mark.asyncio
async def test_the_rail_heading_is_never_cut_mid_word() -> None:
    """AC#1: at the compact rail width the heading ellipsises, never clips."""
    host = _library_host()
    async with host.run_test(size=(100, 30)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        painted = _painted(host, screen.query_one("#library-rail-heading").region)
        assert "Navigati" not in painted or "Navigation" in painted, painted
        assert "Navigat…" in painted or "Navigation" in painted, painted


# --- task-32226: unsubmitted rail text stays on its own canvas -------------


@pytest.mark.asyncio
async def test_unsubmitted_rail_text_never_seeds_the_rag_query_box() -> None:
    """AC#1: text typed into the rail box on a browse canvas is never
    committed to the Search/RAG query state, so it cannot reappear in the
    Search/RAG query box on the next visit."""
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-media").press()
        await _wait_for_selector(screen, pilot, "#library-media-filter")
        box = screen.query_one("#library-search-input", Input)
        box.focus()
        await pilot.press(*"draft")  # typed, never submitted
        await pilot.pause()
        assert box.value == "draft", "the keystrokes stay in the widget"
        screen.query_one("#library-row-browse-search").press()
        query_box = await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        assert query_box.value == "", query_box.value
        assert screen._rag_search_state.query == "", screen._rag_search_state.query
