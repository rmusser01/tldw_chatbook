"""Tests for the LibraryRail widget."""

from __future__ import annotations

import pytest
from textual.widgets import Button, Input

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.Library.library_rail_state import LibraryRailPreferences
from tldw_chatbook.Library.library_shell_state import LibraryShellState
from tldw_chatbook.Widgets.Library.library_rail import LibraryRail


pytestmark = pytest.mark.asyncio


def _make_shell() -> LibraryShellState:
    """Return a minimal Library shell state for rail tests."""
    return LibraryShellState(
        header_line="Library | Test",
        sections=(),
        details_lines=(),
        selected_row_id="",
        canvas_kind="empty",
        canvas_target="",
        canvas_empty_copy="",
    )


async def test_library_rail_top_action_factory(widget_pilot):
    """The top_action_factory is stored and its widgets are rendered first."""
    factory = lambda: [Button("Ingest", id="library-top-action")]
    preferences = LibraryRailPreferences()

    async with await widget_pilot(
        LibraryRail,
        shell=_make_shell(),
        preferences=preferences,
        top_action_factory=factory,
    ) as pilot:
        rail = pilot.app.test_widget
        assert rail.top_action_factory is factory

        await pilot.pause()
        assert isinstance(pilot.app.query_one("#library-top-action", Button), Button)
        assert isinstance(pilot.app.query_one("#library-search-input", Input), Input)
