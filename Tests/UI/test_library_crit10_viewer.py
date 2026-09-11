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


# --------------------------------------------------------------------------
# task-32348: Find
# --------------------------------------------------------------------------


def test_find_is_refused_on_the_tabs_that_have_no_search_bar():
    from tldw_chatbook.Library.library_media_viewer_state import (
        analysis_find_unavailable_reason,
    )

    for mode in ("info", "highlights"):
        assert analysis_find_unavailable_reason(
            mode=mode, analysis="anything", generating=False, editing=False
        ) == "This tab has no text to search · switch to Read or Analysis.", mode
    assert analysis_find_unavailable_reason(
        mode="read", analysis="", generating=False, editing=False
    ) == ""
    assert analysis_find_unavailable_reason(
        mode="analysis", analysis="", generating=False, editing=False
    ) == "No analysis to search yet."


async def _open_first_media_reader(host, pilot):
    """Open the first media item's Reader and return the settled screen."""
    screen = await _open_media_list(host, pilot)
    await _open_first_reader_row(screen, pilot)
    return screen


@pytest.mark.asyncio
async def test_ctrl_f_opens_the_reader_find_bar_and_the_footer_names_it():
    host = _media_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        assert ("ctrl+f", "find") in screen._library_footer_shortcuts_for_current_state()
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert screen.query("#library-media-content-search-controls")
        assert screen._media_state.find_open is True


@pytest.mark.asyncio
async def test_t_never_arms_the_trash_while_find_is_open():
    host = _media_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert screen.check_action("library_media_move_to_trash", ()) is False
        assert ("t", "trash") not in screen._library_footer_shortcuts_for_current_state()
        await pilot.press("escape")
        await pilot.pause()
        assert screen.check_action("library_media_move_to_trash", ()) is True


@pytest.mark.asyncio
async def test_find_is_refused_on_the_info_tab_and_t_stays_a_plain_character():
    """task-32348 AC#2, B D4/D4a: the Info tab has no bar to mount."""
    host = _media_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        screen.query_one("#library-media-reader-select-info", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-mode-info")
        await pilot.pause()
        assert screen.check_action("library_media_reader_find", ()) is False
        assert ("ctrl+f", "find") not in (
            screen._library_footer_shortcuts_for_current_state()
        )
        await pilot.press("ctrl+f")
        await pilot.pause()
        assert screen._media_state.find_open is False
        assert not screen.query("#library-media-content-search-controls")
# --------------------------------------------------------------------------
# task-32365: a Markdown analysis renders
# --------------------------------------------------------------------------


_MARKDOWN_ANALYSIS = "## Key contributions\n\nA first point, and a second."


def _analysis_host(analysis: str) -> LibraryProductionCSSHarness:
    """Media whose stored analysis is ``analysis``.

    Local media detail never carries ``analysis_content`` at the top level;
    the viewer reads the newest ``versions`` entry.
    """
    app = _build_media_test_app()
    items = _two_media_items()
    for item in items:
        item["versions"] = [{"version_number": 1, "analysis_content": analysis}]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


@pytest.mark.asyncio
async def test_a_stored_analysis_renders_its_markdown_with_a_raw_toggle():
    host = _analysis_host(_MARKDOWN_ANALYSIS)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await _switch_to_analysis(screen, pilot)
        body = await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        painted = _painted(host, body.region)
        assert "Key contributions" in painted, painted
        assert "## Key" not in painted, painted
        toggle = screen.query_one("#library-media-analysis-content-mode-raw", Button)
        toggle.press()
        await pilot.pause()
        await pilot.pause()
        assert "## Key contributions" in _painted(
            host, screen.query_one("#library-media-viewer-content").region
        )


@pytest.mark.asyncio
async def test_a_plain_text_analysis_is_offered_no_toggle():
    """Nothing to render, so no affordance for it (the Read tab's rule)."""
    host = _analysis_host("A flat paragraph of prose, with no markup at all.")
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_media_reader(host, pilot)
        await _switch_to_analysis(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-media-viewer-content")
        assert not screen.query("#library-media-analysis-content-mode-raw")
