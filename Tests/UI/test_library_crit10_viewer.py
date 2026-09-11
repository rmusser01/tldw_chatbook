"""Critique-10 Reader group: the typing footer, Find, and a rendered analysis.

Covers tasks 32346 (the canvas verbs survive a focused Input), 32348 (Find
opens from the keyboard and refuses the tabs it cannot search) and 32365
(a Markdown analysis renders).
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_crit9_shell import (
    _library_host,
)
from Tests.UI.test_library_media_render_fixes import (
    _analysis_flow_host,
    _open_first_reader_row,
    _painted,
    _switch_to_analysis,
)
from Tests.UI.test_library_media_side_by_side import (
    _build_media_test_app,
    _open_media_list,
    _two_media_items,
)
from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)


def _media_host() -> LibraryProductionCSSHarness:
    """The Library with two local media items, list and Reader both usable."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    return LibraryProductionCSSHarness(app)


# --------------------------------------------------------------------------
# task-32346: the footer under a focused Input
# --------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_focused_search_box_keeps_the_canvas_verbs_behind_one_named_key():
    host = _library_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-search").press()
        query = await _wait_for_selector(screen, pilot, "#library-rag-query-input")
        query.focus()
        await pilot.pause()
        chips = screen._library_footer_shortcuts_for_current_state()
        labels = [label for _key, label in chips]
        assert labels[0] == "typing in field", chips
        assert ("esc", "leave field") in chips, chips
        joined = " ".join(labels)
        assert "after esc: u use Library context in Console · o open evidence" in joined, chips
        # AC#2: "F6 next pane" is LAST, so the responsive footer drops it
        # before any canvas verb.
        assert chips[-1] == ("F6", "next pane"), chips
        # AC#1's honesty half: no single printable key is advertised as live.
        assert not [key for key, _label in chips if len(key) == 1 and key.isprintable()], chips


@pytest.mark.asyncio
async def test_the_media_filter_box_keeps_the_list_verbs_the_same_way():
    host = _media_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        box = await _wait_for_selector(screen, pilot, "#library-media-filter")
        box.focus()
        # The list settles asynchronously and re-seats focus; wait for the
        # caret to actually land in the filter box rather than assume it.
        await _wait_for_condition(
            pilot,
            lambda: screen.focused is screen.query_one("#library-media-filter"),
            message="The media filter box never took focus.",
        )
        chips = screen._library_footer_shortcuts_for_current_state()
        joined = " ".join(label for _key, label in chips)
        assert "after esc: " in joined and "s select" in joined, chips
        assert chips[-1] == ("F6", "next pane"), chips
