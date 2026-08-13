"""Mounted behavior tests for the Library media-content search controls.

Each test protects the structural recompose boundary: changing active search
state must replace controls, while changing match data must preserve them.
"""

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Input, Static

from tldw_chatbook.Widgets.Library.library_media_content import (
    LibraryMediaContentSearchControls,
)


class SearchControlsHarness(App[None]):
    """Mount controls with an active Markdown query."""

    def compose(self) -> ComposeResult:
        yield LibraryMediaContentSearchControls(
            is_markdown=True,
            query="budget",
            matches=(2, 8),
            match_index=0,
            id="controls",
        )


class BlankSearchControlsHarness(App[None]):
    """Mount controls with no active query."""

    def compose(self) -> ComposeResult:
        yield LibraryMediaContentSearchControls(
            is_markdown=True,
            query="",
            matches=(),
            match_index=0,
            id="controls",
        )


@pytest.mark.asyncio
async def test_match_index_sync_preserves_navigation_identity_and_focus() -> None:
    """Catch a match-index update that recompiles the navigation controls."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)
        previous = app.query_one("#library-media-content-search-prev", Button)
        next_button = app.query_one("#library-media-content-search-next", Button)
        next_button.focus()

        controls.sync_match_index(matches=(2, 8), match_index=1)
        await pilot.pause()

        assert app.query_one("#library-media-content-search-prev") is previous
        assert app.query_one("#library-media-content-search-next") is next_button
        assert app.focused is next_button
        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "Match 2 of 2 matches"
        )


@pytest.mark.asyncio
async def test_active_query_sync_preserves_search_and_navigation_identity() -> None:
    """Catch active-query updates that needlessly recompose mounted controls."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)
        search_input = app.query_one("#library-media-content-search", Input)
        previous = app.query_one("#library-media-content-search-prev", Button)
        next_button = app.query_one("#library-media-content-search-next", Button)

        controls.sync_query_state(
            is_markdown=True,
            query="cost",
            matches=(4,),
            match_index=0,
        )
        await pilot.pause()

        assert app.query_one("#library-media-content-search", Input) is search_input
        assert app.query_one("#library-media-content-search-prev", Button) is previous
        assert app.query_one("#library-media-content-search-next", Button) is next_button
        assert search_input.value == "cost"
        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "Match 1 of 1 matches"
        )


@pytest.mark.asyncio
async def test_match_index_sync_displays_no_matches() -> None:
    """Catch a formatter that leaves a stale match count for an empty result."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)

        controls.sync_match_index(matches=(), match_index=0)
        await pilot.pause()

        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "No matches"
        )


@pytest.mark.asyncio
async def test_match_index_sync_wraps_status_index() -> None:
    """Catch a formatter that exposes an out-of-range rather than wrapped index."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)

        controls.sync_match_index(matches=(2, 8), match_index=3)
        await pilot.pause()

        assert str(app.query_one("#library-media-content-search-status", Static).renderable) == (
            "Match 2 of 2 matches"
        )


@pytest.mark.asyncio
async def test_blank_query_sync_removes_active_search_controls() -> None:
    """Catch a blank-query transition that leaves stale controls mounted."""
    app = SearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)

        controls.sync_query_state(
            is_markdown=True,
            query="",
            matches=(),
            match_index=0,
        )
        await pilot.pause()

        assert not app.query("#library-media-content-search-status")
        assert not app.query("#library-media-content-search-prev")
        assert not app.query("#library-media-content-search-next")


@pytest.mark.asyncio
async def test_active_query_sync_mounts_controls_and_markdown_placeholder() -> None:
    """Catch a blank-to-active transition that fails to recompose its controls."""
    app = BlankSearchControlsHarness()
    async with app.run_test() as pilot:
        controls = app.query_one("#controls", LibraryMediaContentSearchControls)
        blank_input = app.query_one("#library-media-content-search", Input)

        controls.sync_query_state(
            is_markdown=True,
            query="cost",
            matches=(4,),
            match_index=0,
        )
        await pilot.pause()

        active_input = app.query_one("#library-media-content-search", Input)
        assert active_input is not blank_input
        assert active_input.placeholder == "Search content (raw text)…"
        assert app.query_one("#library-media-content-search-prev", Button)
        assert app.query_one("#library-media-content-search-next", Button)
