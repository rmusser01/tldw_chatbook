"""Critique #8 polish contracts for the Library Media/Conversations surfaces.

Covers tasks 32060, 32065, 32067, 32068, 32070 and 32074 -- keyboard select
mode beside a loaded Reader, the below-64-column Media stage, the
conversation reader's identity/timestamps/pager, the Markdown notice and
byline, the footer, and the Prompts variables glyph.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from Tests.UI.test_library_media_side_by_side import (
    NARROW_SIZE,
    WIDE_SIZE,
    _open_media_list,
)
from Tests.UI.test_library_media_reader_flow import _flow_app, _load_row_0
from Tests.UI.test_library_media_toolbar_adapt import _CanvasApp, _select_state
from Tests.UI.test_library_shell import LibraryProductionCSSHarness
from tldw_chatbook.Utils.adaptive_reader_state import ITEMS_MIN_WIDTH
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas


def _painted(widget) -> str:
    """Return exactly the cells ``widget`` occupies on the composited screen."""
    screen = widget.screen
    strips = list(screen._compositor.render_strips())
    region = widget.region
    return "\n".join(
        strips[y].text[region.x : region.right]
        for y in range(region.y, min(region.bottom, len(strips)))
    )


@pytest.mark.asyncio
async def test_s_enters_select_mode_from_an_items_row_beside_a_loaded_reader():
    """task-32060 AC#1: "s" belongs to the focused Items row, not the layout.

    Critique #8: with an item open in the Reader the ``s`` gate read the
    Reader's exit availability, so in every layout that keeps a real exit
    (100x30: Library collapsed, Items beside the Reader) the key was inert
    from a list row and the footer dropped the chip -- forcing the mouse.
    """
    app, service = _flow_app(count=3)
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=NARROW_SIZE) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)

        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        assert ("s", "select") in screen._library_footer_shortcuts_for_current_state()

        await pilot.press("s")
        await pilot.pause()
        assert screen._media_state.select_mode is True
        footer = screen._library_footer_shortcuts_for_current_state()
        assert ("s", "done selecting") in footer
        assert ("space", "toggle selection") in footer

        for media_id in tuple(service.detail_release):
            service.release(media_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("width", [ITEMS_MIN_WIDTH, ITEMS_MIN_WIDTH + 2])
async def test_bulk_delete_confirm_copy_wraps_at_the_items_floor(width: int) -> None:
    """task-32060 AC#2: the safety sentence wraps inside the narrowest pane.

    The canvas floor (36) sat ABOVE the Items pane floor (32), so at the
    pane's narrowest the canvas overflowed its slot and the confirm sentence
    was clipped mid-word at the pane edge ("Delete 2 selected items? You c").
    """
    app = _CanvasApp(_select_state(selected_count=2, confirming=True))
    async with app.run_test(size=(width, 34)) as pilot:
        await pilot.pause()
        canvas = app.query_one("#library-media-canvas", LibraryMediaCanvas)
        copy = app.query_one("#library-media-bulk-delete-confirm-copy", Static)
        assert canvas.region.width <= width
        assert copy.region.right <= width
        painted = " ".join(_painted(copy).split())
        assert (
            "Delete 2 selected items? You can undo right away, "
            "or restore later from Trash." in painted
        ), painted
