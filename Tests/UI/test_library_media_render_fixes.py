"""Render fixes from the 2026-09-03 re-critique (tasks 31221, 31222).

Two verified rendering bugs, both only reproducible on the REAL screen path
(the library's split stylesheet loads lazily via LibraryScreen.CSS_PATH, and
the chooser bug needs the screen's focus-on-open):

- task-31221: the app-global ``*:focus`` solid outline (core/_reset.tcss)
  paints OVER a widget's outermost rows without costing geometry; the
  screen focuses the option-count-height type chooser on open, so with the
  common two-option catalogue the outline covered every option — an empty
  bordered band, selection blind. Third widget bitten after TASK-1160
  (DataTable) and TASK-2300 (SelectOverlay).
- task-31222: ``#library-media-reader-mode-read`` had no height rule (an
  unstyled Vertical defaulted to 1fr, holding a blank band above the Find
  bar) while ``#library-media-viewer-content`` was capped at 18 rows
  regardless of terminal size.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, OptionList

from Tests.UI.test_library_media_side_by_side import (
    _build_media_test_app,
    _open_media_list,
    _two_media_items,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_selector,
)


def _host() -> LibraryProductionCSSHarness:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    return LibraryProductionCSSHarness(app)


def _painted(host, region) -> str:
    strips = list(host.screen._compositor.render_strips())
    return "\n".join(
        strips[y].crop(region.x, region.right).text
        for y in range(region.y, min(region.bottom, len(strips)))
    )


@pytest.mark.asyncio
async def test_type_chooser_paints_every_option():
    """Every option's TEXT is painted in the opened chooser (task-31221).

    Painted text on purpose: the bug was the app-global ``*:focus`` solid
    outline painting OVER the option rows without affecting layout, so
    region assertions passed while the user saw an empty bordered band.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-type-filter", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-type-choices")
        await pilot.pause()
        await pilot.pause()
        chooser = screen.query_one("#library-media-type-choices", OptionList)
        assert chooser.option_count >= 2
        assert chooser.has_focus  # the screen focuses the chooser on open
        painted = _painted(host, chooser.region)
        assert "All types" in painted, painted
        assert "video" in painted, painted
        assert "audio" in painted, painted


@pytest.mark.asyncio
async def test_sort_chooser_paints_every_option():
    """task-31235: all four sort options render, as a vertical OptionList.

    Critique #3 P1: the horizontal choice strip clipped "Title A-Z" and
    rendered "Title Z-A" nowhere at the items pane's real width, while
    keyboard selection could still pick the invisible option — an option
    you can't see doesn't exist. Painted text on purpose (the 31221
    lesson): geometry-only assertions are blind to clipping and to
    focus-outline paint-over alike.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen.query_one("#library-media-sort", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-sort-choices")
        await pilot.pause()
        await pilot.pause()
        chooser = screen.query_one("#library-media-sort-choices", OptionList)
        assert chooser.option_count == 4
        assert chooser.has_focus  # the screen focuses the chooser on open
        painted = _painted(host, chooser.region)
        for label in ("Newest", "Oldest", "Title A-Z", "Title Z-A"):
            assert label in painted, painted


async def _open_first_reader_row(screen, pilot):
    screen.query_one("#library-media-row-0", Button).press()
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._library_media_reader_session.pending_request is None
            and screen._library_media_reader_session.loaded_id is not None
        ),
        message="Reader detail never settled.",
    )
    await pilot.pause()


@pytest.mark.asyncio
async def test_reader_content_fills_the_remaining_pane():
    """task-31237 (evolves task-31222's scaled cap into a true fill):
    the content box takes the pane's remaining height exactly — no
    stranded band below it, and the viewer itself never scrolls (chrome
    stays pinned; overflow belongs to the content box's own scroll)."""
    host = _host()
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        mode_read = screen.query_one("#library-media-reader-mode-read")
        # Heading (+ optional toggle) only — never a 1fr blank band.
        assert mode_read.region.height <= 4, mode_read.region
        viewer = screen.query_one("#library-media-viewer")
        content = screen.query_one("#library-media-viewer-content")
        # Fill: the box's bottom reaches the pane bottom (1-row margin).
        assert content.region.bottom >= viewer.region.bottom - 2, (
            content.region,
            viewer.region,
        )
        # No outer scroll: the viewer's virtual height fits its container.
        assert viewer.virtual_size.height <= viewer.container_size.height, (
            viewer.virtual_size,
            viewer.container_size,
        )


@pytest.mark.asyncio
async def test_find_bar_collapsed_until_find_and_escape_recollapses():
    """task-31237: the content Find bar mounts only on the Find action.

    A permanently open "Search content…" input duplicated the Find button
    and spent 3 rows on every fresh item; Escape must collapse it again
    (the old behavior cleared the query but left the bar).
    """
    host = _host()
    async with host.run_test(size=(170, 48)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        assert not screen.query("#library-media-content-search-controls")

        screen.query_one("#library-media-reader-find", Button).press()
        await _wait_for_selector(
            screen, pilot, "#library-media-content-search-controls"
        )
        search_input = await _wait_for_selector(
            screen, pilot, "#library-media-content-search"
        )
        await _wait_for_condition(
            pilot,
            lambda: search_input.has_focus,
            message="Find never focused the search input.",
        )

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-content-search-controls")

        # Qodo on #2367: the bar is a reader substate, focus-agnostic —
        # moving focus OUT of the bar (to the content body) must not
        # strand it; Escape still closes it first.
        screen.query_one("#library-media-reader-find", Button).press()
        await _wait_for_selector(
            screen, pilot, "#library-media-content-search-controls"
        )
        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-content-search-controls")
