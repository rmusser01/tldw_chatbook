"""TASK-22228 items 6-7: reader sub-state presses and the resize layout leg.

Item 6 -- the six Reader button presses that flip one viewer sub-state
(Edit metadata / Cancel, Move to trash / Cancel, Edit analysis / Cancel)
were still whole-screen recomposes, tearing down the nav bar, footer, rail
and canvas for a flip that changes only the mounted ``LibraryMediaViewer``'s
children. Escape already made the identical three flips through the
task-21116 seam (``_sync_library_media_viewer_or_recompose``); these tests
pin that the BUTTONS take the same route -- measured 1 whole-screen
recompose per press before, 0 after, for all six.

Item 7 -- the screen-level ``Resize`` leg scheduled the Media reader layout
resolve on every Library resize frame, including routes where
``#library-media-reader-shell`` is not mounted at all, where it could only
walk the whole Library DOM to find nothing (a FAILED ``query_one`` takes no
id-cache fast path). Measured 2 such calls per resize on the Conversations
route; the on-route path must keep working, which the second arm pins.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from textual.widgets import Button

import tldw_chatbook.UI.Screens.library_screen as library_screen_module
from Tests.UI.test_library_canvas_scoped_sync import _screen_recompose_spy
from Tests.UI.test_library_per_click_recompose_t21116 import _media_app_host
from Tests.UI.test_library_shell import (
    LIBRARY_TEST_SIZE,
    _active_library_screen,
    _wait_for_library_shell,
    _wait_for_selector,
)
from tldw_chatbook.UI.Navigation.base_app_screen import BaseAppScreen
from tldw_chatbook.Widgets.Library.library_media_viewer import LibraryMediaViewer


async def _boot_media_library(host, pilot):
    """Mount the Library shell and open Browse Media.

    The starter rail's "Explore all tools" disclosure is only composed for
    an undisclosed lifecycle, so it is pressed only when present -- this
    harness resolves the full rail directly on some fixtures.
    """
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    if screen.query("#library-rail-explore-all"):
        screen.query_one("#library-rail-explore-all", Button).press()
    await _wait_for_selector(screen, pilot, "#library-row-browse-media")
    screen.query_one("#library-row-browse-media", Button).press()
    await _wait_for_selector(screen, pilot, "#library-media-row-0")
    return screen


async def _open_reader_with_more_actions(screen, pilot):
    """Open the first media item's Reader and disclose the More action list."""
    screen.query_one("#library-media-row-0", Button).press()
    await _wait_for_selector(screen, pilot, "#library-media-reader-more")
    screen.query_one("#library-media-reader-more", Button).press()
    await _wait_for_selector(screen, pilot, "#library-media-edit")


@pytest.mark.asyncio
async def test_reader_substate_presses_are_viewer_scoped() -> None:
    """All six sub-state presses rebuild the viewer, never the whole screen."""
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        await _open_reader_with_more_actions(screen, pilot)
        rail_before = screen.query_one("#library-rail")
        viewer_before = screen.query_one("#library-media-viewer", LibraryMediaViewer)

        recomposes: dict[str, int] = {}

        async def press(selector: str, settles: str) -> None:
            calls, spy = _screen_recompose_spy()
            with patch.object(BaseAppScreen, "refresh", spy):
                screen.query_one(selector, Button).press()
                await _wait_for_selector(screen, pilot, settles)
                await pilot.pause()
            recomposes[selector] = len(calls)

        # Move to trash: arm the confirm, then cancel it.
        await press("#library-media-delete", "#library-media-delete-confirm")
        assert screen._library_media_confirming_delete is True
        await press("#library-media-delete-cancel", "#library-media-delete")
        assert screen._library_media_confirming_delete is False

        # Edit metadata: enter the Info-mode form, then cancel it.
        await press("#library-media-edit", "#library-media-edit-cancel")
        assert screen._library_media_editing is True
        assert screen._library_media_reader_session.mode == "info"
        await press("#library-media-edit-cancel", "#library-media-reader-select-read")
        assert screen._library_media_editing is False

        # Edit analysis: the analysis mode owns its own edit/cancel pair.
        screen.query_one("#library-media-reader-select-analysis", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")
        await press("#library-media-analysis-edit", "#library-media-analysis-cancel")
        assert screen._library_media_editing_analysis is True
        await press("#library-media-analysis-cancel", "#library-media-analysis-edit")
        assert screen._library_media_editing_analysis is False

        assert len(recomposes) == 6, recomposes
        assert all(count == 0 for count in recomposes.values()), recomposes
        # Viewer-scoped: the rail and the viewer NODE both survive; only the
        # viewer's children were rebuilt.
        assert screen.query_one("#library-rail") is rail_before
        assert (
            screen.query_one("#library-media-viewer", LibraryMediaViewer)
            is viewer_before
        )


@pytest.mark.asyncio
async def test_resize_off_the_media_route_does_not_resolve_the_media_layout() -> None:
    """A Conversations resize frame never looks for the Media reader shell."""
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        screen.query_one("#library-row-browse-conversations", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-reader-shell")

        calls = {"count": 0}
        original = type(screen)._sync_library_media_reader_layout_from_shell

        def counting(self, *args, **kwargs):
            calls["count"] += 1
            return original(self, *args, **kwargs)

        with patch.object(
            type(screen), "_sync_library_media_reader_layout_from_shell", counting
        ):
            width, height = LIBRARY_TEST_SIZE
            await pilot.resize_terminal(width - 2, height)
            await pilot.pause()
            await pilot.pause()

        assert calls["count"] == 0, calls


@pytest.mark.asyncio
async def test_resize_on_the_media_route_still_carries_the_focus_intent() -> None:
    """Control arm: the guard scopes the leg, it does not retire it.

    Counting layout resolves alone cannot see this leg at all -- the
    shell's own ``AdaptiveReaderShellResized`` message resolves the layout
    too, so deleting the screen-level leg outright leaves a resolve-count
    assertion green (it did, when this arm was first written). What is
    unique to the screen-level leg is the pre-refresh ``focus_intent``
    capture: it is the ONLY caller that passes one, and it is what lets a
    pane closing on a resize hand focus to its grip instead of stranding it
    on a hidden widget. So the arm counts calls CARRYING that intent.
    """
    host = _media_app_host()
    async with host.run_test(size=LIBRARY_TEST_SIZE) as pilot:
        screen = await _boot_media_library(host, pilot)
        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-more")
        shell = screen.query_one("#library-media-reader-shell")
        screen.query_one("#library-media-reader-more", Button).focus()
        await pilot.pause()
        assert screen.focused is not None

        with_intent = {"count": 0}
        original = type(screen)._sync_library_media_reader_layout_from_shell

        def counting(self, *args, **kwargs):
            if kwargs.get("focus_intent") is not None:
                with_intent["count"] += 1
            return original(self, *args, **kwargs)

        with patch.object(
            type(screen), "_sync_library_media_reader_layout_from_shell", counting
        ):
            width, height = LIBRARY_TEST_SIZE
            await pilot.resize_terminal(width - 6, height)
            await pilot.pause()
            await pilot.pause()

        assert with_intent["count"] >= 1, with_intent
        # ...and the resize really did settle the mounted shell's geometry.
        assert shell.effective_layout == screen._library_media_reader_layout
