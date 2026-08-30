"""The Context rail's default layout must fit the viewport it is given.

TASK-23193. A 2026-08-29 UX audit measured the shipped default across ten
terminal geometries and found it overflowed every one of them -- including
200x60, a full-screen terminal on a 27" display. At 160x48 the rail rendered
51 rows into a 32-row viewport: 19 rows (37%) below the fold, with three of
seven sections entirely invisible on a fresh install.

The defect is a default-configuration choice, not a rendering bug: five of
seven sections ship open. These tests pin the outcome a user experiences --
"the rail I am given fits the space it has" -- rather than any particular
row budget, so a later change that reclaims rows elsewhere is free to open
more sections again.
"""

from __future__ import annotations

import pytest
from textual.containers import VerticalScroll
from textual.widgets import Static

from Tests.UI.test_console_left_rail import make_console_pilot


async def _rail_geometry(pilot):
    """Return (viewport_rows, content_rows, hint_visible) for the Context rail."""
    screen = pilot.app.screen
    outer = screen.query_one("#console-left-rail-body", VerticalScroll)
    try:
        hint = screen.query_one("#console-left-rail-outer-hint", Static)
        hint_visible = bool(hint.display)
    except Exception:
        hint_visible = False
    return outer.size.height, outer.virtual_size.height, hint_visible


@pytest.mark.asyncio
async def test_default_context_rail_fits_a_standard_terminal() -> None:
    """160x48 is an ordinary maximised laptop terminal; the rail must fit it."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        await pilot.pause(0.3)
        viewport, content, hint_visible = await _rail_geometry(pilot)

        assert viewport > 0, "Context rail viewport did not resolve"
        assert content <= viewport, (
            f"Context rail wants {content} rows but has {viewport}; "
            f"{content - viewport} rows are below the fold on a fresh install"
        )
        assert not hint_visible, (
            "the outer overflow hint is showing, so the default still overflows"
        )


@pytest.mark.asyncio
async def test_default_open_sections_are_sessions_and_conversations() -> None:
    """The default expands only the two sections a user navigates by."""
    from tldw_chatbook.Chat.console_rail_state import ConsoleRailPreferences

    defaults = ConsoleRailPreferences()
    open_sections = {
        "session": defaults.session_open,
        "workspace": defaults.workspace_open,
        "conversations": defaults.conversations_open,
        "model": defaults.model_open,
        "agent": defaults.agent_open,
        "details": defaults.details_open,
        "character": defaults.character_open,
    }
    assert {name for name, is_open in open_sections.items() if is_open} == {
        "session",
        "conversations",
    }, f"unexpected default open set: {open_sections}"


@pytest.mark.asyncio
async def test_every_context_section_header_is_reachable_without_scrolling() -> None:
    """A user must be able to see that a collapsed section exists at all.

    The 2026-08-29 audit's sharpest finding was not that content was clipped
    but that whole sections were invisible: a fresh install gave no hint that
    Agent, Details or Character existed. Headers are one row each, so all
    seven fitting is a much weaker requirement than all seven being open.
    """
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        await pilot.pause(0.3)
        screen = pilot.app.screen
        outer = screen.query_one("#console-left-rail-body", VerticalScroll)
        top = outer.region.y
        bottom = top + outer.size.height

        hidden = []
        for section_id in (
            "session",
            "workspace",
            "conversations",
            "model",
            "agent",
            "details",
            "character",
        ):
            header = screen.query_one(f"#console-rail-section-header-{section_id}")
            if not (header.display and top <= header.region.y < bottom):
                hidden.append(section_id)

        assert not hidden, f"sections invisible without scrolling: {hidden}"


@pytest.mark.asyncio
@pytest.mark.parametrize("columns", [117, 120, 128, 129])
async def test_context_survives_the_former_inspector_auto_open_band(columns) -> None:
    """TASK-23197: 118..128 used to delete the Context rail without asking.

    The Inspector auto-opened itself in that band, which tripped
    ``resolve_console_rail_priority`` and force-collapsed Context to a
    13-column stub. A one-column resize from 117 to 118 swapped which
    sidebar the user had, with no explanation. Nothing about the user's
    preferences changed across that boundary, so nothing about which rail
    they can see should either.
    """
    async with make_console_pilot(size=(columns, 40), production_styles=True) as pilot:
        await pilot.pause(0.4)
        screen = pilot.app.screen

        rail = screen.query_one("#console-left-rail")
        assert rail.display, (
            f"the Context rail is gone at {columns} columns with default "
            "preferences; the auto-open band is evicting it again"
        )
