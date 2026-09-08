"""Critique-8 polish/shell fixes for the Library landing, rail and Notes.

Covers tasks 32058, 32059, 32061, 32062, 32063, 32064, 32066, 32069, 32071
and 32072 -- the polish-shell group of the critique-8 fix wave.
"""

from __future__ import annotations

import pytest

from tldw_chatbook.Widgets.Library import LibraryLandingCanvas
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_library_shell,
)

#: The compact geometry the critique-8 live review ran at (register row 17).
COMPACT_TEST_SIZE = (100, 30)
WIDE_TEST_SIZE = (170, 48)


@pytest.mark.asyncio
async def test_library_landing_canvas_is_hidden_at_compact_widths():
    """task-32066: library.md says the rail owns navigation below 120 columns.

    At 100x30 the landing canvas ("Search everything…", counts, From your
    Library, Quick actions) was still painted beside the 22-column rail.
    """
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=COMPACT_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.pause()

        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)
        rail = screen.query_one("#library-rail")

        assert rail.display is True
        assert landing.region.width == 0, (
            "the landing canvas must not paint at compact widths"
        )


@pytest.mark.asyncio
async def test_library_landing_canvas_paints_at_wide_widths():
    """The same route keeps the landing beside the rail above the breakpoint."""
    app = _build_test_app()
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE_TEST_SIZE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await pilot.pause()

        landing = screen.query_one("#library-landing-canvas", LibraryLandingCanvas)

        assert landing.region.width > 0
