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

import asyncio
import dataclasses
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from types import SimpleNamespace
from textual.widgets import Button, Input, OptionList, Static, TextArea
from textual.worker import WorkerState

from tldw_chatbook.Library.ingest_analysis import NO_ANALYSIS_PROVIDER_NEXT_STEP
from tldw_chatbook.Library.library_media_reader_state import set_mode, set_more_open
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.Library.library_media_state import (
    library_media_int_backing_id,
)
from tldw_chatbook.UI.Screens.library_screen import _sync_library_canvas
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Library.library_adaptive_reader_shell import (
    LIBRARY_ADAPTIVE_READER_GRIP_CLASS,
)
from tldw_chatbook.Widgets.Library.library_browse_reader_shell import (
    LibraryBrowseReaderShell,
)

from Tests.UI.test_library_media_side_by_side import (
    _active_library_screen,
    _build_media_test_app,
    _open_media_list,
    _two_media_items,
    _wait_for_library_shell,
)
from Tests.UI.test_library_media_reader_flow import (
    ControlledDetailMediaService,
    _load_row_0,
    _many_media_items,
    _row_identity,
    _wait_for_detail_call,
)
from Tests.UI.test_library_shell import (
    LibraryGlobalKeyProductionCSSHarness,
    LibraryProductionCSSHarness,
    _open_media_find,
    _painted_cells,
    _row_is_painted_focused,
    _seed_conversations,
    _submit_content_search_query,
    _top_border_row,
    _two_conversations,
    _wait_for_condition,
    _wait_for_selector,
)


#: task-31981: the full surfaced reason for the no-provider case, derived
#: from the source's next-step constant so only the reason half is pinned
#: here (the "reason · action" join and next step live in the source).
_NO_PROVIDER_REASON = (
    f"No analysis provider is configured · {NO_ANALYSIS_PROVIDER_NEXT_STEP}."
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
            screen._media_state.reader_session.pending_request is None
            and screen._media_state.reader_session.loaded_id is not None
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


# ---------------------------------------------------------------------------
# task-31269 (critique #4 P0): the Find gesture, not the mount, decides focus.
# ---------------------------------------------------------------------------


def _analysis_flow_host(count: int = 3):
    """Three local items, each with a current analysis version.

    Local media detail never carries ``analysis_content`` at the top level;
    the viewer reads the newest ``versions`` entry
    (``library_media_viewer_state._latest_version_analysis_text``).
    """
    app = _build_media_test_app()
    items = _many_media_items(count)
    for index, item in enumerate(items, 1):
        item["versions"] = [
            {"version_number": 1, "analysis_content": f"Analysis of item {index}"}
        ]
    _seed_conversations(app, _two_conversations(), media=items)
    service = ControlledDetailMediaService(items)
    app.media_reading_scope_service = service
    return LibraryProductionCSSHarness(app), service


async def _walk_next(screen, service, pilot, expected_row: int) -> str:
    """Press ] and settle the Reader on ``expected_row``; return its id."""
    row = screen.query_one(f"#library-media-row-{expected_row}", Button)
    row_id, backing_id, _ = _row_identity(row)
    await pilot.press("right_square_bracket")
    await _wait_for_detail_call(service, backing_id)
    service.release(backing_id)
    await _wait_for_condition(
        pilot,
        lambda: screen._media_state.reader_session.loaded_id == row_id,
        message=f"] never loaded row {expected_row}.",
    )
    await pilot.pause()
    return row_id


async def _switch_to_analysis(screen, pilot) -> None:
    screen._media_state.reader_session = set_mode(
        screen._media_state.reader_session, "analysis"
    )
    screen._sync_library_media_viewer_or_recompose()
    await _wait_for_selector(screen, pilot, "#library-media-reader-mode-analysis")
    await pilot.pause()


@pytest.mark.asyncio
async def test_analysis_mode_walk_never_moves_focus_into_the_search_field():
    """task-31269 (critique #4 P0): ] in Analysis mode walks, it never types.

    #2367's focus-on-mount hook fired on EVERY mount with an empty query,
    and the Analysis tab (task-28026) mounted the bar unconditionally, so
    each item load in Analysis mode parked focus in the Input and the next
    ] became text (live: `▊ ]`, `]]]]]`).
    """
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        await _switch_to_analysis(screen, pilot)
        # The bar is collapsed until Find asks for it, exactly like Read.
        assert not screen.query("#library-media-content-search-controls")

        await _walk_next(screen, service, pilot, expected_row=1)
        assert screen._media_state.reader_session.mode == "analysis"
        assert not isinstance(screen.focused, Input), screen.focused
        assert not screen.query("#library-media-content-search-controls")

        # A second ] must still be a walk (the P0 symptom was it being typed).
        await _walk_next(screen, service, pilot, expected_row=2)
        assert not isinstance(screen.focused, Input), screen.focused


@pytest.mark.asyncio
async def test_find_on_the_analysis_tab_opens_the_bar_there_and_escape_closes_it():
    """Find searches what you are reading: on Analysis it opens the analysis
    bar (task-28026's Analysis->Read jump is retired), focuses its Input, and
    one Escape collapses it (live: the first Escape only blurred)."""
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        await _switch_to_analysis(screen, pilot)
        screen.query_one("#library-media-reader-find", Button).press()
        search_input = await _wait_for_selector(
            screen, pilot, "#library-media-content-search"
        )
        await _wait_for_condition(
            pilot,
            lambda: search_input.has_focus,
            message="Find never focused the analysis search input.",
        )
        assert screen._media_state.reader_session.mode == "analysis"

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-content-search-controls")
        assert screen._media_state.find_open is False


@pytest.mark.asyncio
async def test_read_mode_walk_with_an_empty_find_bar_never_steals_focus():
    """task-31269 AC2: an open, still-empty bar survives an item change, but
    focus stays where the reader left it, so ] keeps walking.

    The empty-query remount was the Read-mode face of the P0: Find opened,
    nothing typed yet, focus moved to the content body, then ] -- the
    remounted bar took the caret and the next ] was typed. A submitted
    query surviving traversal is pinned separately by
    test_a_new_document_rescans_for_the_same_query.
    """
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        screen.query_one("#library-media-reader-find", Button).press()
        search_input = await _wait_for_selector(
            screen, pilot, "#library-media-content-search"
        )
        await _wait_for_condition(
            pilot, lambda: search_input.has_focus, message="Find never focused."
        )
        # Leave the field the way a reader does (F6 target = content body).
        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        assert not isinstance(screen.focused, Input)

        await _walk_next(screen, service, pilot, expected_row=1)
        assert screen._media_state.find_open is True
        assert screen.query("#library-media-content-search-controls")
        assert not isinstance(screen.focused, Input), screen.focused

        await _walk_next(screen, service, pilot, expected_row=2)
        assert not isinstance(screen.focused, Input), screen.focused


@pytest.mark.asyncio
async def test_find_toggles_the_bar_closed_when_it_is_open():
    """task-31269 AC4: a second Find press closes the bar (live: it did nothing)."""
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        screen.query_one("#library-media-reader-find", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-content-search-controls")
        screen.query_one("#library-media-reader-find", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-content-search-controls")
        assert screen._media_state.find_open is False


# ---------------------------------------------------------------------------
# task-31270 (critique #4 P1): receipts paint Undo and Dismiss at pane width.
# ---------------------------------------------------------------------------


def _items_pane_width(screen) -> int:
    return screen.query_one("#library-media-canvas").region.width


@pytest.mark.asyncio
async def test_delete_receipt_paints_undo_and_dismiss_at_the_items_pane_width():
    """task-31270 (critique #4 P1): the receipt's Undo was clipped to `Und`
    in the ~38-col Items pane (live cap_99). Painted text on purpose: a
    region assertion cannot see a label cut by its parent's width."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.delete_receipt_ids = ("local:media:1",)
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-bulk-delete-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        assert receipt.region.width <= _items_pane_width(screen)
        painted = _painted(host, receipt.region)
        assert "✓ deleted · 1 item · in Trash" in painted, painted
        assert "Undo" in painted, painted
        assert "Dismiss" in painted, painted


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
async def test_delete_receipt_paints_a_live_undo_on_a_stale_page(size):
    """task-31220 (critique #5): the receipt painted "✓ deleted · 1 item
    · in Trash" beside a DISABLED "○ Undo" because the page behind it
    had gone stale -- the confirmation's "You can undo right away" broken
    at the moment it mattered. Undo restores the ids the receipt itself
    names, so the stale PAGE cannot invalidate it. Painted text on
    purpose: the "○" marker is a label change a region assertion cannot
    see."""
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller
        controller.freshness = "stale"
        controller.stale_copy = "Media changed; retry to load a current page."
        screen._media_state.delete_receipt_ids = ("local:media:1",)
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-bulk-delete-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        assert receipt.region.width <= _items_pane_width(screen)
        painted = _painted(host, receipt.region)
        assert "\u2713 deleted \u00b7 1 item \u00b7 in Trash" in painted, painted
        assert "Undo" in painted, painted
        assert "Dismiss" in painted, painted
        assert "\u25cb" not in painted, painted
        assert (
            screen.query_one("#library-media-bulk-delete-undo", Button).disabled
            is False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
async def test_failed_undo_receipt_paints_its_reason_and_retry(size):
    """task-31220: the ✗ state is the one this task added, so it gets the same
    painted probe the ✓ state has. task-31270 clipped "Undo" to "Und" at the
    Items pane's real width and only a painted assertion saw it; "Retry undo"
    is four cells longer, and a reason can run to 80 chars."""
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.delete_receipt_ids = ("local:media:1",)
        screen._media_state.delete_receipt_undo_failure = (
            "1 of 2 \u00b7 database is locked"
        )
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-bulk-delete-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        assert receipt.region.width <= _items_pane_width(screen)
        painted = _painted(host, receipt.region)
        # The copy row is ``width: 100%; height: auto``, so a long reason wraps
        # DOWNWARD rather than clipping or pushing the actions row off -- which
        # is the behaviour this probe exists to confirm. Whitespace is collapsed
        # across that wrap; clipping still fails it, because clipped characters
        # do not come back.
        flat = " ".join(painted.split())
        assert "\u2717 undo failed \u00b7 1 of 2 \u00b7 database is locked" in flat, painted
        # Unwrapped on its own row: "Retry undo" is four cells longer than the
        # "Undo" that task-31270 saw clipped to "Und" at this width.
        assert "Retry undo" in painted, painted
        assert "Dismiss" in painted, painted
        assert "\u2713 deleted" not in flat, painted
        assert "\u25cb" not in painted, painted


@pytest.mark.asyncio
async def test_dismiss_receipt_paints_undo_at_the_items_pane_width():
    """task-31270: the set-dismiss receipt clipped to `… Un` (live cap_83)."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._review_dismiss_receipt_name = lambda: "2 selected items"
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-review-dismiss-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        assert receipt.region.width <= _items_pane_width(screen)
        painted = _painted(host, receipt.region)
        assert "✓ dismissed · 2 selected items" in painted, painted
        assert "Undo" in painted, painted
        assert "Dismiss" in painted, painted


@pytest.mark.asyncio
async def test_analyze_receipt_paints_its_counts_retry_and_dismiss():
    """task-28007 AC#4: the bulk-Analyze receipt is PR A's two-row grammar --
    honest counts plus the two actions, readable in the ~38-col Items pane."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.analyze_total = 40
        screen._media_state.analyze_done = 38
        screen._media_state.analyze_failed_ids = ("local:media:1", "local:media:2")
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-analyze-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        assert receipt.region.width <= _items_pane_width(screen)
        painted = _painted(host, receipt.region)
        assert "\u2713 analyzed \u00b7 38 of 40 \u00b7 2 failed" in painted, painted
        assert "Retry failed" in painted, painted
        assert "Dismiss" in painted, painted


@pytest.mark.asyncio
async def test_analyze_receipt_paints_the_running_copy():
    """Review round 1 (I6): the frozen running copy had no test at all."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.analyze_running = True
        screen._media_state.analyze_total = 40
        screen._media_state.analyze_done = 0
        screen._media_state.analyze_failed_ids = ("local:media:1", "local:media:2")
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-analyze-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        painted = _painted(host, receipt.region)
        assert "Analyzing 3 of 40 \u00b7 2 failed" in painted, painted
        # A run in flight offers neither action: nothing to retry yet, and a
        # Dismiss that did not cancel would lie.
        assert "Retry failed" not in painted, painted
        assert "Dismiss" not in painted, painted


@pytest.mark.asyncio
async def test_analyze_receipt_omits_the_failed_segment_and_retry_at_zero():
    """Review round 1 (I6): the clean-run form -- no "· 0 failed", no Retry."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.analyze_total = 40
        screen._media_state.analyze_done = 40
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-analyze-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        painted = _painted(host, receipt.region)
        assert "\u2713 analyzed \u00b7 40 of 40" in painted, painted
        assert "failed" not in painted, painted
        assert "Retry failed" not in painted, painted
        assert "Dismiss" in painted, painted


@pytest.mark.asyncio
async def test_analyze_receipt_never_ticks_a_run_where_nothing_succeeded():
    """Review round 1 (I5): "\u2713 analyzed \u00b7 0 of 3" was a checkmark over a
    total failure. Retry is still offered -- only the glyph was dishonest."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.analyze_total = 3
        screen._media_state.analyze_done = 0
        screen._media_state.analyze_failed_ids = ("a", "b", "c")
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-analyze-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        painted = _painted(host, receipt.region)
        assert "\u2717 analyzed \u00b7 0 of 3 \u00b7 3 failed" in painted, painted
        assert "\u2713" not in painted, painted
        assert "Retry failed" in painted, painted


@pytest.mark.asyncio
async def test_analyze_overwrite_choice_paints_both_options():
    """task-28007 AC#3: the already-analysed choice is armed IN the receipt row."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.analyze_choice = (
            ("local:media:1", "local:media:2"),
            ("local:media:2",),
        )
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-analyze-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        assert receipt.region.width <= _items_pane_width(screen)
        painted = _painted(host, receipt.region)
        assert "1 of 2 already analyzed" in painted, painted
        assert "—" not in painted, painted  # R3: no dangling dash
        assert "Skip them" in painted, painted
        assert "Overwrite" in painted, painted
        # (final review, I-1) A THIRD button did not fit the Items pane at
        # its 36-cell floor: the row painted "Skip them  Overwrite  Dism".
        # "Skip them" already IS the change-nothing outcome and retires the
        # card, so the choice state offers no Dismiss at all.
        assert "Dism" not in painted, painted
        assert not screen.query("#library-media-analyze-receipt-dismiss")


@pytest.mark.asyncio
async def test_a_scope_change_clears_the_armed_analyze_choice():
    """(final review, I-1/M-3) The choice is a PENDING action over a
    snapshot of ids, armed after select mode already exited -- so nothing
    used to invalidate it. Changing the browse scope (here: the filter)
    must retire it rather than keep offering "Overwrite" over ids the user
    can no longer see."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.analyze_choice = (
            ("local:media:1", "local:media:2"),
            ("local:media:2",),
        )
        _sync_library_canvas(screen, "media")
        await _wait_for_selector(screen, pilot, "#library-media-analyze-receipt")

        screen._request_library_media_filter("beta")
        await pilot.pause()
        assert screen._media_state.analyze_choice is None
        _sync_library_canvas(screen, "media")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-analyze-receipt")


@pytest.mark.asyncio
async def test_an_import_origin_run_paints_no_receipt_on_the_media_canvas():
    """(final review, I-2) An "Analyze N skipped" run drives the SAME
    screen-owned counters, so its progress used to render as a Media
    receipt on a canvas the user never started it from -- whose "Retry
    failed" would then re-run those ids as a media run, leaving the Import
    rows saying "analysis failed" forever. The Import surface has its own
    per-row receipts; the Media canvas must show nothing for that run,
    while it runs and after it settles."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.analyze_origin = "import"
        screen._media_state.analyze_running = True
        screen._media_state.analyze_total = 3
        screen._media_state.analyze_done = 1
        _sync_library_canvas(screen, "media")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-analyze-receipt")

        screen._media_state.analyze_running = False
        screen._media_state.analyze_failed_ids = ("local:media:1", "local:media:2")
        _sync_library_canvas(screen, "media")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-analyze-receipt")
        assert not screen.query("#library-media-analyze-retry")


@pytest.mark.asyncio
async def test_select_mode_bulk_rows_paint_analyze_export_and_review():
    """task-28007 AC#4: Analyze is readable beside (below) Export and Review.

    Measured: the Items pane is 36 cells and Clear/Export/Review already
    take 33 of them, so a fourth 13-cell action on that row clipped every
    label. Analyze therefore rides its own row -- the same multi-row
    grammar Delete already uses -- and this pins BOTH rows readable."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._toggle_library_media_select_mode()
        row = await _wait_for_selector(
            screen, pilot, "#library-media-select-actions"
        )
        analyze_row = screen.query_one("#library-media-select-analyze")
        await pilot.pause()
        await pilot.pause()
        pane = _items_pane_width(screen)
        assert row.region.width <= pane
        assert analyze_row.region.width <= pane
        painted = _painted(host, row.region)
        assert "Export" in painted, painted
        assert "Review" in painted, painted
        analyze_painted = _painted(host, analyze_row.region)
        assert "Analyze" in analyze_painted, analyze_painted


@pytest.mark.asyncio
async def test_pressing_analyze_leaves_select_mode_and_paints_its_receipt():
    """Review round 1 (I4 + the missing end-to-end pin): the REAL button
    press must drop the select-mode toolbar at once (not one partition
    pass later) and the finished run must paint its own receipt -- the
    other painted tests set the canvas fields by hand."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        analyzed = []

        async def _one(media_id, *, resolution):
            analyzed.append(media_id)
            return True

        screen._analyze_one_library_media_item = _one
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                library_screen_module,
                "analysis_unavailable_reason",
                lambda *_a, **_k: "",
            )
            screen._toggle_library_media_select_mode()
            await _wait_for_selector(
                screen, pilot, "#library-media-analyze-selected"
            )
            screen.query_one("#library-media-row-0").press()
            screen.query_one("#library-media-row-1").press()
            await pilot.pause()
            screen.query_one("#library-media-analyze-selected", Button).press()
            await pilot.pause()
            assert not screen.query("#library-media-select-actions")
            assert not screen.query("#library-media-select-analyze")
            receipt = await _wait_for_selector(
                screen, pilot, "#library-media-analyze-receipt"
            )
            await _wait_for_condition(
                pilot,
                lambda: screen._media_state.analyze_running is False,
                message="the run never settled",
            )
            await pilot.pause()
            painted = _painted(host, receipt.region)
        assert len(analyzed) == 2, analyzed
        assert "\u2713 analyzed \u00b7 2 of 2" in painted, painted
        assert "Dismiss" in painted, painted


@pytest.mark.asyncio
async def test_analyze_run_that_dies_with_the_screen_says_where_it_stopped():
    """Review round 1 (I2): the worker is owned by this screen, so Textual
    cancels it on unmount, and navigating back builds a NEW LibraryScreen
    whose receipt fields start empty -- an arbitrary prefix would be
    analysed with nothing on screen ever saying so."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        notices = []
        screen.app_instance.notify = lambda message, **kwargs: notices.append(
            (message, kwargs)
        )
        entered = asyncio.Event()
        release = asyncio.Event()
        cancelled = asyncio.Event()

        async def _one(media_id, *, resolution):
            if media_id != "b":
                return True
            entered.set()
            try:
                await release.wait()
            except asyncio.CancelledError:
                cancelled.set()
                raise
            return True

        screen._analyze_one_library_media_item = _one
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                library_screen_module,
                "analysis_unavailable_reason",
                lambda *_a, **_k: "",
            )
            screen._start_library_media_analyze(("a", "b", "c"), overwrite=True)
        worker = next(
            candidate
            for candidate in host.workers
            if candidate.group
            == library_screen_module._ANALYZE_SELECTED_WORKER_GROUP
        )
        await _wait_for_condition(
            pilot, entered.is_set, message="the run never reached item 2"
        )
        await host.pop_screen()
        await pilot.pause()
        await pilot.pause()

        assert cancelled.is_set(), "the in-flight item must be cancelled"
        assert worker.state is WorkerState.CANCELLED
        assert notices, "a cancelled run must say where it stopped"
        assert notices[0][0] == (
            "Analysis stopped at 1 of 3 · reopen Select ▸ Analyze to "
            "continue; finished items are skipped"
        ), notices
        assert notices[0][1].get("severity") == "warning"


@pytest.mark.asyncio
async def test_analyze_run_cancelled_before_a_total_is_known_says_so_honestly():
    """task-28007 Task 3 (N1): an unmount that lands DURING the AC#3
    partition pass -- before ``_library_media_analyze_total`` is ever
    stamped -- must not notify the nonsensical "stopped at 0 of 0"."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        notices = []
        screen.app_instance.notify = lambda message, **kwargs: notices.append(
            (message, kwargs)
        )
        entered = asyncio.Event()
        release = asyncio.Event()

        async def _blocked_unanalyzed(media_ids):
            entered.set()
            await release.wait()
            return media_ids

        screen._library_media_unanalyzed_ids = _blocked_unanalyzed
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                library_screen_module,
                "analysis_unavailable_reason",
                lambda *_a, **_k: "",
            )
            screen._start_library_media_analyze(("a", "b"), overwrite=False)
        worker = next(
            candidate
            for candidate in host.workers
            if candidate.group
            == library_screen_module._ANALYZE_SELECTED_WORKER_GROUP
        )
        await _wait_for_condition(
            pilot,
            entered.is_set,
            message="the run never reached the partition pass",
        )
        assert screen._media_state.analyze_total == 0
        await host.pop_screen()
        await pilot.pause()
        await pilot.pause()

        assert worker.state is WorkerState.CANCELLED
        assert notices, "a run cancelled before its total is known must still say so"
        assert notices[0][0] == "Analysis stopped before it started", notices
        assert notices[0][1].get("severity") == "warning"


@pytest.mark.asyncio
async def test_analyze_bulk_action_follows_the_selection_in_place():
    """task-28007 AC#4, learning task-28242's Qodo #2335 lesson: the
    in-place row-toggle patcher must flip Analyze alongside Export/Review
    /Delete, or the first checked row leaves it disabled until some
    unrelated recompose. The provider gate still outranks the count."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                library_screen_module,
                "analysis_unavailable_reason",
                lambda *_a, **_k: "",
            )
            screen._toggle_library_media_select_mode()
            await _wait_for_selector(
                screen, pilot, "#library-media-analyze-selected"
            )
            assert (
                screen.query_one("#library-media-analyze-selected", Button).disabled
                is True
            )
            screen.query_one("#library-media-row-1").press()
            await pilot.pause()
            analyze = screen.query_one("#library-media-analyze-selected", Button)
            assert analyze.disabled is False, "checked rows must arm Analyze"
            assert "Analyze" in str(analyze.label)

            # And the provider gate wins over the count: an unready
            # provider keeps it off with its own reason.
            mp.setattr(
                library_screen_module,
                "analysis_unavailable_reason",
                lambda *_a, **_k: "No analysis provider is configured.",
            )
            screen._media_state.analyze_reason_cache = None
            screen.query_one("#library-media-row-0").press()
            await pilot.pause()
            gated = screen.query_one("#library-media-analyze-selected", Button)
            assert gated.disabled is True
            assert gated.tooltip == "No analysis provider is configured."


# ---------------------------------------------------------------------------
# task-31271 -- the footer told the truth at four seams
# ---------------------------------------------------------------------------


def _footer_labels(screen) -> list[str]:
    return [
        label for _key, label in screen._library_footer_shortcuts_for_current_state()
    ]


def _painted_footer(host, screen) -> str:
    return _painted(host, screen.query_one(AppFooterStatus).region)


@pytest.mark.asyncio
async def test_footer_drops_close_find_after_escape_closes_the_bar():
    """Seam (a): Escape closes Find, so the esc chip must stop saying so.

    ``_library_media_escape_label`` asked the DOM whether the Find bar was
    mounted, and the read happened BEFORE the recompose that removes it --
    so the footer kept promising "esc close find" over a closed bar (A cap
    08/23, B cap_21) while Escape actually focused the Items pane.
    task-31272 shortened that chip to the shared "close".
    """
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        screen.query_one("#library-media-reader-find", Button).press()
        search_input = await _wait_for_selector(
            screen, pilot, "#library-media-content-search"
        )
        await _wait_for_condition(
            pilot, lambda: search_input.has_focus, message="Find never focused."
        )
        # The chip is genuinely visible at this width before Escape, so its
        # absence afterwards cannot be footer compaction.
        assert "close" in _footer_labels(screen)
        assert "close" in _painted_footer(host, screen)

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        assert screen._media_state.find_open is False
        labels = _footer_labels(screen)
        assert "close" not in labels, labels
        painted = _painted_footer(host, screen)
        assert "close" not in painted, painted


@pytest.mark.asyncio
async def test_pressing_s_focuses_a_media_row_so_space_toggles_immediately():
    """Seam (b): "s" must land focus on a row, or the Space chip is a lie.

    Entering select mode advertised "space toggle selection" while Space
    was a no-op until the user hunted for a row (A cap 31->32, B cap_69).
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()

        assert screen._media_state.select_mode is True
        assert ("space", "toggle selection") in (
            screen._library_footer_shortcuts_for_current_state()
        )
        focused = screen.focused
        assert focused is not None and focused.has_class(
            "library-media-row"
        ), focused

        await pilot.press("space")
        await pilot.pause()
        assert screen._media_state.row_selection.count == 1


@pytest.mark.asyncio
async def test_space_in_select_mode_never_reaches_the_pane_grip():
    """Seam (b), second half: Space in select mode belongs to the selection.

    With focus on the Library pane grip (a Button, so its own "space"
    binding resolves before the screen's) Space COLLAPSED the Library pane
    while the footer said "toggle selection" -- B cap_69.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        assert screen._media_state.select_mode is True

        shell = screen.query_one(
            ".library-media-route", LibraryBrowseReaderShell
        )
        before = shell.effective_layout
        shell.library_grip.focus()
        await pilot.pause()

        await pilot.press("space")
        await pilot.pause()
        await pilot.pause()

        shell = screen.query_one(
            ".library-media-route", LibraryBrowseReaderShell
        )
        assert shell.effective_layout.library_open is before.library_open
        assert screen._media_state.row_selection.count == 0


@pytest.mark.asyncio
async def test_reader_footer_advertises_l_c_t():
    """Seam (d): l / c / t are real Reader keys and t ARMS DELETE (B cap_97).

    All three shipped ``show=False`` and appeared in no footer set, so the
    only way to discover the key that moves an item to Trash was to press
    it.
    """
    host, service = _analysis_flow_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _load_row_0(screen, service, pilot)
        shortcuts = screen._library_footer_shortcuts_for_current_state()
        keys = [key for key, _label in shortcuts]
        assert "l" in keys, shortcuts
        assert "c" in keys, shortcuts
        assert "t" in keys, shortcuts
        # Painted, not just computed: the three chips are gated on the
        # SETTLED session, and the footer registered at open time (while
        # the detail request was still in flight) advertised none of them.
        # Live, they only appeared after an unrelated F6.
        painted = _painted_footer(host, screen)
        assert "l read later" in painted, painted
        assert "c use in Console" in painted, painted
        assert "t trash" in painted, painted


@pytest.mark.asyncio
async def test_space_in_select_mode_is_claimed_from_rows_and_grips_only():
    """Seam (b), third half: Space is taken from the GRIP, not from every
    button.

    The priority binding outranks the focused widget, so the gate has to
    claim exactly what Space should own: a media row (toggle it) and the
    reader shell's pane grips (chrome -- swallow the collapse the footer
    never promised). "Done" and the other select-mode buttons keep their
    own key handling; asserted on the gate itself because Textual 8's
    ``Button.BINDINGS`` is ``enter`` only -- Space does nothing on a plain
    Button either way, so a "press Done with Space" assertion would pass
    without the narrowing. Enter still presses Done, end to end.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        assert screen._media_state.select_mode is True

        def claim() -> bool | None:
            return screen.check_action("library_media_toggle_row_selection", ())

        # Entry lands on a row: Space is the selection key there.
        assert screen.focused.has_class("library-media-row"), screen.focused
        assert claim() is True

        shell = screen.query_one(
            ".library-media-route", LibraryBrowseReaderShell
        )
        shell.library_grip.focus()
        await pilot.pause()
        assert claim() is True  # swallowed, so the pane never collapses

        done = screen.query_one("#library-media-select-toggle", Button)
        assert str(done.label) == "Done"
        done.focus()
        await pilot.pause()
        assert claim() is False  # the button keeps its own key handling

        await pilot.press("enter")
        await pilot.pause()
        await pilot.pause()
        assert screen._media_state.select_mode is False


# ---------------------------------------------------------------------------
# task-31272 (critique #4 P1): Escape, F6 and Back never strand focus in a
# text input, and the Reader's exit ladder matches the layout on screen.
# ---------------------------------------------------------------------------

#: The whole Reader Escape vocabulary. Critique #4 saw eight distinct
#: ``esc …`` chips live; every state now lands on one of these four (or on
#: no chip at all, where Escape genuinely does nothing).
_ESCAPE_LABELS = {"close", "focus Items", "focus Library", "back"}


def _global_key_host() -> LibraryGlobalKeyProductionCSSHarness:
    """``_host()`` with TldwCli's real global F6 binding attached."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    return LibraryGlobalKeyProductionCSSHarness(app)


def _media_layout(screen):
    return screen.query_one(
        ".library-media-route", LibraryBrowseReaderShell
    ).effective_layout


@pytest.mark.asyncio
async def test_escape_from_the_reader_lands_on_the_loaded_row_then_the_rail_row():
    """Escape walks Reader -> loaded row -> rail row, never into an Input.

    Critique #4 (A cap 20): leaving the Reader took three Escapes and the
    second landed inside "Search Library…", so the third was typed text.
    The Reader also stays LIVE on the way out (the three-pane shell has no
    "list mode" of its own), so ] / [ keep working from the Items row.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        layout = _media_layout(screen)
        assert layout.items_open and layout.library_open, layout

        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is not None
        assert screen.focused.has_class("library-media-row"), screen.focused
        assert screen._media_state.view == "viewer"
        assert screen.check_action("library_media_next_item", ()) is True

        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is not None
        assert not isinstance(screen.focused, Input), screen.focused
        assert screen.focused.id == "library-row-browse-media", screen.focused


@pytest.mark.asyncio
async def test_escape_closes_the_more_menu_from_any_reader_focus():
    """Escape closes More wherever focus sits inside the Reader region.

    Critique #4 (B cap_106): the footer promised "close more" and Escape
    did nothing. Traced to the stale view flag -- "‹ Back" (and the rail's
    Escape) set ``_media_state.view = "list"`` while the three-pane
    Reader kept painting the document, and every Reader binding is gated
    on that flag, so Escape/]/[ died on identical pixels.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        # The rail's own Escape must not strand the flag either.
        screen.query_one("#library-row-browse-media").focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert screen._media_state.view == "viewer"
        # The terminus leaves Escape UN-GATED rather than swallowing it
        # (the Conversations seam): no chip, no action, no strand.
        assert screen._library_media_escape_label() == ""
        assert screen.check_action("library_media_viewer_back", ()) is False

        screen.query_one("#library-media-reader-more", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-more-actions")
        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert screen._media_state.reader_session.more_open is False
        assert not screen.query("#library-media-reader-more-actions")


@pytest.mark.asyncio
async def test_f6_content_stop_is_visible_and_content_still_paints():
    """The F6 Reader stop tints the content box's own border (task-31221).

    Critique #4 (B cap_57): F6's first Reader candidate is the content
    scroller, but nothing on screen said so. The cue must be the EXISTING
    border, never an outline: the app-global ``*:focus`` outline paints
    OVER the outermost rows, which is why this asserts painted text too.
    task-31634 upgraded the tint to a heavy border (glyphs, not colour
    alone); the sibling test below diffs the painted border row.
    """
    host = _global_key_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        box = screen.query_one("#library-media-viewer-content")
        unfocused = box.styles.border_top
        screen.query_one("#library-search-input", Input).focus()
        await pilot.pause()

        for _ in range(6):
            await pilot.press("f6")
            await pilot.pause()
            if str(getattr(screen.focused, "id", "")) in {
                "library-media-viewer-content-text",
                "library-media-viewer-content",
            }:
                break
        assert str(getattr(screen.focused, "id", "")) in {
            "library-media-viewer-content-text",
            "library-media-viewer-content",
        }, screen.focused

        box = screen.query_one("#library-media-viewer-content")
        focused_border = box.styles.border_top
        # task-31634: the cue is now the GLYPHS as well as the colour --
        # heavy when focused, the unchanged solid otherwise. Still a
        # one-cell border either way, so no reading row moves.
        assert unfocused[0] == "solid", unfocused
        assert focused_border[0] == "heavy", focused_border
        assert focused_border[1] != unfocused[1], (focused_border, unfocused)
        painted = _painted(host, box.region)
        assert "product demo video walks through" in painted.lower(), painted


@pytest.mark.asyncio
async def test_back_button_is_not_composed_in_the_side_by_side_layout():
    """"‹ Back" only renders where a "back to list" exit is real.

    Critique #4 (A cap 25/31/39): in the three-pane shell the Items pane
    already shows the list, so Back changed no pixels -- it only flipped
    the view flag out from under the visible Reader.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        layout = _media_layout(screen)
        assert layout.items_open and layout.library_open, layout
        assert not screen.query("#library-media-back")

    compact = _host()
    async with compact.run_test(size=(100, 30)) as pilot:
        screen = await _open_media_list(compact, pilot)
        await _open_first_reader_row(screen, pilot)
        layout = _media_layout(screen)
        assert not (layout.items_open and layout.library_open), layout
        assert screen.query("#library-media-back")


@pytest.mark.asyncio
async def test_escape_labels_are_one_of_four():
    """Every Reader state's ``esc`` chip comes from one four-word set."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        seen: dict[str, str] = {}

        def record(state: str) -> None:
            label = screen._library_media_escape_label()
            seen[state] = label
            assert label in _ESCAPE_LABELS or label == "", (state, label)

        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        record("plain Reader")

        screen.query_one("#library-media-reader-find", Button).press()
        await _wait_for_selector(
            screen, pilot, "#library-media-content-search-controls"
        )
        record("find open")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        screen.query_one("#library-media-reader-more", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-more-actions")
        record("more open")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        await pilot.press("t")
        await _wait_for_selector(screen, pilot, "#library-media-delete-cancel")
        record("armed delete")
        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()

        screen.query_one("#library-media-row-0", Button).focus()
        await pilot.pause()
        record("items focus")

        screen.query_one("#library-row-browse-media").focus()
        await pilot.pause()
        record("rail focus")

        assert seen["plain Reader"] == "focus Items", seen
        assert seen["find open"] == "close", seen
        assert seen["more open"] == "close", seen
        assert seen["armed delete"] == "close", seen
        assert seen["items focus"] == "focus Library", seen
        # The rail row is the ladder's terminus: Escape does nothing
        # there, so the footer advertises no chip rather than a "back"
        # the three-pane shell cannot perform.
        assert seen["rail focus"] == "", seen


@pytest.mark.asyncio
async def test_escape_to_the_row_restores_the_list_keys_beside_the_reader():
    """The Items pane keeps its own keys while the three-pane Reader is open.

    task-31272 review: with "‹ Back" gone, ``_library_media_view`` never
    leaves "viewer" for the whole visit, so everything keyed on that flag
    (the ``s`` select gate, the list/select footer sets, the type/sort
    strips) would stay dead while the user is standing on a list row.
    They key on the live list SURFACE instead -- the Items region holding
    focus -- and hand the keys straight back on F6 into the Reader.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        assert screen.check_action("library_media_toggle_select_mode", ()) is False
        assert dict(screen._library_footer_shortcuts_for_current_state())["esc"] == (
            "focus Items"
        )

        await pilot.press("escape")
        await pilot.pause()
        assert screen.focused is not None
        assert screen.focused.has_class("library-media-row"), screen.focused
        assert screen.check_action("library_media_toggle_select_mode", ()) is True
        # The Reader keeps the footer from a list row -- ] / l / c / t are
        # all still live from there -- and "s" joins it through its gate.
        row_labels = _footer_labels(screen)
        assert "select" in row_labels, row_labels
        assert "read later" in row_labels, row_labels

        await pilot.press("s")
        await pilot.pause()
        await pilot.pause()
        assert screen._media_state.select_mode is True
        # Select mode is the Items pane genuinely taking the keys.
        assert screen._library_footer_shortcuts_for_current_state() == (
            screen.LIBRARY_MEDIA_SELECT_SHORTCUTS
        )

        # Leaving select mode puts the Reader's own set back -- REGISTERED,
        # not just computed: the ] / [ chips derive from the mounted rows,
        # and a registration mid-swap dropped them while the keys worked.
        await pilot.press("s")
        await _wait_for_condition(
            pilot,
            lambda: (
                screen._media_state.select_mode is False
                and "next item"
                in [label for _key, label in screen._footer_shortcut_registration[1]]
            ),
            message="Leaving select mode left a stale footer registration.",
        )

        # F6 back into the Reader hands the Reader its own set again.
        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        labels = _footer_labels(screen)
        assert "focus Items" in labels, labels
        assert "toggle selection" not in labels, labels


@pytest.mark.asyncio
async def test_escape_cancels_an_items_choice_strip_over_the_reader():
    """A type/sort strip opened beside the Reader owns Escape, as advertised."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        screen.query_one("#library-media-type-filter", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-type-choices")
        await pilot.pause()
        assert dict(screen._library_footer_shortcuts_for_current_state())["esc"] == (
            "cancel"
        )

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-type-choices")
        assert screen._media_state.view == "viewer"


@pytest.mark.asyncio
async def test_escape_cancels_a_strip_that_is_open_while_the_reader_has_focus():
    """Qodo on #2386: strip VISIBILITY is not the same question as key ownership.

    The open-strip check had picked up the Items region's focus gate, so a
    strip left open while focus moved into the Reader was invisible to
    Escape (which hopped focus instead of closing it) and to the chip that
    describes Escape.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        screen.query_one("#library-media-sort", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-sort-choices")
        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        assert dict(screen._library_footer_shortcuts_for_current_state())["esc"] == (
            "close"
        )

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-sort-choices")
        assert screen._media_state.view == "viewer"

@pytest.mark.asyncio
async def test_find_is_disabled_with_a_reason_when_the_analysis_tab_has_nothing_to_search():
    """Qodo on #2378: with Find now opening the bar for the tab being read,
    an Analysis tab with no analysis (or one still generating) has no bar
    to mount -- pressing Find must not become a silent state toggle. The
    button says why it is off, and the handler refuses to arm find_open."""
    host = _host()  # _two_media_items carry no analysis
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        await _switch_to_analysis(screen, pilot)
        find = screen.query_one("#library-media-reader-find", Button)
        assert find.disabled is True
        assert "No analysis" in str(find.tooltip)
        # Belt and braces: the handler itself refuses.
        screen.handle_library_media_reader_find(SimpleNamespace(stop=lambda: None))
        await pilot.pause()
        await pilot.pause()
        assert screen._media_state.find_open is False
        assert not screen.query("#library-media-content-search-controls")


# ---------------------------------------------------------------------------
# Wave 4 PR C (tasks 31276 / 31277 / 31274) -- merged after PR B landed.
# ---------------------------------------------------------------------------


def _find_host() -> LibraryProductionCSSHarness:
    """A reader item whose body carries several matches for "item"."""
    app = _build_media_test_app()
    items = _two_media_items()
    for item in items:
        item["content"] = "\n".join(
            f"Line {number} mentions the item." for number in range(1, 9)
        )
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


def _painted_row(host, y: int) -> str:
    """Return the whole painted screen row at ``y`` (pane join included)."""
    strips = list(host.screen._compositor.render_strips())
    return strips[y].text


@pytest.mark.asyncio
async def test_find_bar_keeps_its_place_through_submit_and_next():
    """task-31276 (critique #4 P2): submitting must not relocate the bar.

    task-15774 docked an ACTIVE search to the top of the viewer, so Enter
    teleported the whole bar from under the mode row to above the Reader
    header and shoved that header down six rows (live cap_20). The bar's
    anchor is the mode row, at every stage of the gesture: open, submit,
    match navigation.

    The header row this pins used to be the "Local Media item" identity
    line; task-31277 made that line server-only, so the title -- now the
    Reader header's own top text row -- is the anchor.
    """
    host = _find_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        await _open_media_find(screen, pilot)
        controls = screen.query_one("#library-media-content-search-controls")
        mode_row = screen.query_one("#library-media-reader-mode-toolbar")
        header = screen.query_one("#library-media-viewer-title")
        opened_y = controls.region.y
        assert mode_row.region.y < opened_y
        assert header.region.y < opened_y

        await _submit_content_search_query(screen, pilot, "item")
        status = screen.query_one("#library-media-content-search-status")
        assert "Match 1 of" in _painted(host, status.region), _painted(
            host, status.region
        )
        assert screen.query_one(
            "#library-media-content-search-controls"
        ).region.y == opened_y
        assert (
            screen.query_one("#library-media-reader-mode-toolbar").region.y
            == mode_row.region.y
        )
        assert (
            screen.query_one("#library-media-viewer-title").region.y
            == header.region.y
        )

        screen.query_one("#library-media-content-search-next", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one(
            "#library-media-content-search-controls"
        ).region.y == opened_y


@pytest.mark.asyncio
async def test_no_join_artifact_after_find_closes():
    """task-31276 (critique #4 P2): no `┐─────` run at the pane join.

    After Escape closed Find, a five-cell rule appeared in the pane-grip
    columns immediately left of the Reader header and persisted across
    later interactions (14 live captures; absent on a fresh open). It is
    the focused grip's accent end-caps: the grip is as tall as the shell,
    so an outline can only paint its FIRST and LAST rows.

    Sampled over the Reader's first three rows on purpose. task-31277
    removed the identity line, so the pane's first row is now `‹ Back` and
    the title is the second -- a single-row sample on the title is blind to
    the outline-top, which paints on the first row (proved: restoring
    `outline-top`/`outline-bottom` on the grip leaves a title-row-only
    assertion passing).
    """
    host = _find_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)

        def join_slices() -> list[str]:
            """The grip's own columns left of the Reader, top three rows."""
            viewer = screen.query_one("#library-media-viewer")
            grip = screen.query_one("#library-browse-items-grip")
            title = screen.query_one("#library-media-viewer-title")
            # The sample must actually cover the header: Back, title, toolbar.
            assert title.region.y - viewer.region.y <= 2, (
                title.region,
                viewer.region,
            )
            # task-31272 (PR B) removed "‹ Back" from the side-by-side layout,
            # so the pane's first row is the title row itself; the sample
            # still starts at the viewer's top edge, where the grip paints.
            return [
                # task-31633 AC#2: the grip is one column now, not five, so
                # the sample is anchored to the grip itself -- a literal five
                # reaches back into the Items pane and reads its border.
                _painted_row(host, y)[grip.region.x : grip.region.right]
                for y in range(viewer.region.y, viewer.region.y + 3)
            ]

        async def settled_join() -> list[str]:
            await pilot.pause()
            await pilot.pause()
            return join_slices()

        fresh_join = join_slices()
        assert not any("─" in row for row in fresh_join), fresh_join
        title = screen.query_one("#library-media-viewer-title")
        assert "Product Demo Video" in _painted_row(host, title.region.y)

        # Find opened then closed. The rule lands in the grip columns, so
        # the header's own region cannot see it -- the join is the assertion.
        await _open_media_find(screen, pilot)
        await pilot.press("escape")
        after_find = await settled_join()
        assert not any("─" in row for row in after_find), after_find
        assert after_find == fresh_join, (fresh_join, after_find)

        # A mode-tab click.
        screen.query_one("#library-media-reader-select-analysis", Button).press()
        after_tab = await settled_join()
        assert not any("─" in row for row in after_tab), after_tab
        assert after_tab == fresh_join, (fresh_join, after_tab)

        # The More menu opened then dismissed.
        screen.query_one("#library-media-reader-more", Button).press()
        await pilot.pause()
        await pilot.press("escape")
        after_more = await settled_join()
        assert not any("─" in row for row in after_more), after_more
        assert after_more == fresh_join, (fresh_join, after_more)


def _plain_local_host() -> LibraryProductionCSSHarness:
    """Two local items with neither an author nor a URL.

    ``_two_media_items`` carries an author on both rows, so it can never
    show the empty byline row task-31277 collapses. The content is plain
    prose with no Markdown marker, so no Rendered|Raw strip can enter the
    chrome count (since task-32234 the content sniff decides that alone,
    for every media type), and one deliberately long line proves the
    reading measure.
    """
    app = _build_media_test_app()
    long_line = (
        "The recorded discussion ran long and this single unbroken sentence "
        "exists to prove that the reading measure wraps the body well before "
        "the full width of a 235 column terminal ever gets used up by prose."
    )
    items = [
        {
            "id": f"media-{index}",
            "title": f"Roadmap Recording {index}",
            # task-32234: the content above sniffs as plain prose, so no
            # Rendered|Raw strip can appear whatever this type is.
            "type": "pdf",
            "last_modified": "2026-07-06T08:00:00Z",
            "keywords": ["roadmap"],
            "content": "\n".join(
                [f"Line 1 of recording {index}.", long_line]
                + [f"Line {number} of recording {index}." for number in range(2, 40)]
            ),
            "version": 1,
        }
        for index in (1, 2)
    ]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


@pytest.mark.asyncio
async def test_local_reader_chrome_stops_before_the_sixth_row():
    """task-31277 (critique #4 P2): nine rows of chrome before the first
    content line (measured live at 235x52). The identity line restates what the Media list already
    said, the byline row paints empty when an item has no author or URL,
    and the section header repeats the selected mode tab. Counted from the
    reader pane's top edge to the first content line, inclusive of the
    content box's top border: Back, title, toolbar, mode row, border."""
    host = _plain_local_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        viewer = screen.query_one("#library-media-viewer")
        body = screen.query_one("#library-media-viewer-content-text")
        chrome = body.region.y - viewer.region.y
        painted = _painted(host, viewer.region)
        assert chrome <= 5, (chrome, painted.splitlines()[:10])
        # An identity line only a server item needs, and a byline row with
        # nothing to say, are simply not composed.
        assert not screen.query("#library-media-reader-identity")
        assert not screen.query("#library-media-reader-byline")
        assert "Local Media item" not in painted, painted
        assert "Roadmap Recording 1" in painted, painted
        assert "Line 1 of recording 1." in painted, painted


@pytest.mark.asyncio
async def test_reader_bodies_do_not_repeat_the_selected_mode_tab():
    """task-31277 AC#3: the mode row is the label; a `Read`/`Analysis`
    section header directly beneath it spent a row of the reading surface
    saying the same word twice."""
    host = _plain_local_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        read_body = screen.query_one("#library-media-reader-mode-read")
        assert not read_body.query(".destination-section"), list(
            read_body.query(".destination-section")
        )
        assert "Read" not in _painted(
            host, screen.query_one("#library-media-viewer-content").region
        )

        screen.query_one("#library-media-reader-select-analysis", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-mode-analysis")
        await pilot.pause()
        assert not screen.query("#library-media-viewer-analysis-title")
        # Re-queried after the recompose: the press replaces these widgets.
        analysis_body = screen.query_one("#library-media-reader-mode-analysis")
        mode_row = screen.query_one("#library-media-reader-mode-toolbar")
        assert analysis_body.region.y == mode_row.region.bottom, (
            analysis_body.region,
            mode_row.region,
        )


@pytest.mark.asyncio
async def test_reader_body_wraps_at_a_reading_measure():
    """task-31277 AC#4: prose ran ~150 cells at 235 columns, against
    DESIGN.md's 65-75. The body caps at ~90 cells; the bordered box keeps
    the full pane width so its border still spans the pane."""
    host = _plain_local_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        box = screen.query_one("#library-media-viewer-content")
        body = screen.query_one("#library-media-viewer-content-text")
        # The box spans the full pane while the text retains its reading
        # measure. Wider navigation and five-cell controls reduce the pane's
        # spare columns without changing the prose cap.
        work = screen.query_one(".library-adaptive-reader-work")
        assert box.region.width == work.region.width, (box.region, work.region)
        assert work.region.width > 92, work.region
        assert body.region.width <= 92, (body.region, box.region)
        # Painted proof the wrap index was built at the capped width: the
        # long line's tail lands on the row below it, not off at column 150.
        rows = _painted(host, body.region).splitlines()
        assert "The recorded discussion ran long" in rows[1], rows[:4]
        assert "terminal ever gets used up by prose." not in rows[1], rows[:4]


def _transcript_host() -> LibraryProductionCSSHarness:
    """Two video items whose transcripts are sectioned with `##` headings."""
    app = _build_media_test_app()
    items = [
        {
            "id": f"media-{index}",
            "title": f"Product Demo {index}",
            "type": "video",
            "last_modified": "2026-07-06T10:00:00Z",
            "content": (
                "## Section 1\n\nThe host opens the demo.\n\n"
                "## Section 2\n\nThe dashboard walkthrough begins.\n"
            ),
            "version": 1,
        }
        for index in (1, 2)
    ]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


@pytest.mark.asyncio
async def test_video_transcript_headings_render_instead_of_painting_hashes():
    """task-31277 AC#5: `_is_markdown_media` gated the content sniff on a
    media-type allowlist that excluded video/audio, so a transcript whose
    sections are `## Section 1` painted the hashes literally."""
    host = _transcript_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        assert screen.query("#library-media-viewer-content-markdown"), list(
            screen.query_one("#library-media-viewer-content").children
        )
        painted = _painted(
            host, screen.query_one("#library-media-viewer-content").region
        )
        assert "Section 1" in painted, painted
        assert "##" not in painted, painted


def _keyword_media_items() -> list[dict[str, object]]:
    """Rows whose keyword appears in NO title and NO body (task-31274)."""
    return [
        {
            "id": "media-1",
            "title": "Opening remarks",
            "type": "article",
            "last_modified": "2026-07-06T08:00:00Z",
            "keywords": ["day2"],
            "content": "Transcript of the opening remarks session.",
            "version": 1,
        },
        {
            "id": "media-2",
            "title": "Closing remarks",
            "type": "article",
            "last_modified": "2026-07-06T10:00:00Z",
            "keywords": ["day3"],
            "content": "Transcript of the closing remarks session.",
            "version": 1,
        },
    ]


@pytest.mark.asyncio
async def test_reprojection_skips_rows_the_page_does_not_retain():
    """task-31961: a bulk Analyze over a multi-page selection reads ONE page.

    ``has_analysis`` is re-read per saved item, and every read costs an
    id-scoped SELECT. An item outside the retained page has no mounted row
    to repaint, so its read can only ever be thrown away -- the membership
    test that decides that belongs ABOVE the fetch, not after it.
    """
    host = _review_state_host(count=24, analysed=0)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        for _ in range(3):
            await pilot.pause()

        retained = {
            str(item["id"])
            for item in screen._library_media_browse_controller.retained_items
        }
        assert len(retained) == 20, retained
        off_page = [
            media_id
            for media_id in (f"local:media:{index}" for index in range(1, 25))
            if media_id not in retained
        ]
        assert len(off_page) == 4, off_page
        on_page = sorted(retained)[:2]

        service = host.app_instance.media_reading_scope_service
        searches_before = len(service.search_calls)
        # The selection spans both pages, exactly as a bulk Analyze over a
        # "select all" does.
        for media_id in [*on_page, *off_page]:
            await screen._reproject_library_media_analysis_row(media_id)

        extra = service.search_calls[searches_before:]
        assert len(extra) == len(on_page), extra
        allowlists = [call["id_allowlist"] for call in extra]
        assert allowlists == [
            [library_media_int_backing_id(media_id)] for media_id in on_page
        ], allowlists


async def _apply_media_filter(screen, pilot, query: str) -> None:
    """Type into the Items filter and wait for the browse scope to apply."""
    screen.query_one("#library-media-filter", Input).value = query
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._library_media_browse_controller.applied_scope is not None
            and screen._library_media_browse_controller.applied_scope.query == query
        ),
        message=f"The media filter never applied query {query!r}.",
    )
    await pilot.pause()


@pytest.mark.asyncio
async def test_media_filter_matches_a_keyword_absent_from_title_and_body():
    """task-31274: a keyword-tagged row is found by its keyword alone.

    ``day2`` is a keyword on exactly one seeded row and appears in no title
    and no body, so a hit is provably keyword-driven.
    """
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_keyword_media_items())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "day2")

        titles = [row.title for row in screen._build_library_media_state().rows]
        assert titles == ["Opening remarks"], titles
        painted = _painted(host, screen.query_one("#library-media-list").region)
        assert "Opening remarks" in painted, painted
        assert "Closing remarks" not in painted, painted


@pytest.mark.asyncio
async def test_media_filter_miss_names_the_fields_it_searched():
    """task-31274 AC#3: the empty state says what the filter searched."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_keyword_media_items())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "zz")

        status = screen.query_one("#library-media-status", Static)
        assert str(status.renderable) == (
            "No media matched “zz” in titles, content or keywords."
        )
        assert not screen.query_one(
            "#library-media-filter-clear", Button
        ).disabled


@pytest.mark.asyncio
async def test_media_filter_placeholder_names_the_fields_it_searches():
    """task-31274 AC#2: the input says keywords are matched too."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_keyword_media_items())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        placeholder = screen.query_one("#library-media-filter", Input).placeholder
        # Short on purpose: Textual paints only the first wrapped line of a
        # placeholder, and the default Items pane fits ~15 cells (task-31274).
        assert placeholder == "Title/keyword…"
        assert len(placeholder) <= 15


# ---------------------------------------------------------------------------
# Wave 4 PR D (task-28007 AC#5) -- Generate says why it is off.
# ---------------------------------------------------------------------------


def _analysed_host() -> LibraryProductionCSSHarness:
    """Two local items, each already carrying a current analysis version.

    Drives the "Regenerate" spelling of the same action, which is gated by
    the same reason and would otherwise go unpinned.
    """
    app = _build_media_test_app()
    items = _two_media_items()
    for index, item in enumerate(items, 1):
        item["versions"] = [
            {"version_number": 1, "analysis_content": f"Analysis of item {index}"}
        ]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


@pytest.mark.parametrize(
    ("host_factory", "expected_label"),
    [(_host, "○ Generate"), (_analysed_host, "○ Regenerate")],
)
@pytest.mark.asyncio
async def test_generate_is_disabled_with_its_reason_when_no_provider_is_configured(
    host_factory, expected_label
):
    """Critique #4 P1: the Reader's Generate learned that no analysis
    provider is configured only AFTER the click, as a toast. It now wears
    the resolver's own reason, in the ``○``-with-reason grammar PR A gave
    Find -- and the post-click guard still refuses with the same words.
    Both spellings of the action ("Generate" / "Regenerate") are gated."""
    host = host_factory()  # the test config configures no analysis provider
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        await _switch_to_analysis(screen, pilot)
        generate = screen.query_one("#library-media-analysis-generate", Button)
        assert generate.disabled is True
        assert str(generate.label) == expected_label
        # The marker is not just on the widget -- it reaches the glass.
        assert expected_label in _painted(host, generate.region)
        assert str(generate.tooltip) == _NO_PROVIDER_REASON
        # Belt and braces: the handler refuses with the same sentence.
        warnings: list[str] = []
        screen._notify_library_media_analysis_warning = warnings.append
        screen.handle_library_media_analysis_generate(
            SimpleNamespace(stop=lambda: None)
        )
        await pilot.pause()
        assert warnings == [_NO_PROVIDER_REASON]
        assert screen._media_state.generating_analysis is False


@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
@pytest.mark.asyncio
async def test_reader_generate_reason_is_painted_inline_not_hover_only(size):
    """The blocked Generate's reason paints inline, not as hover-only chrome.

    task-31981 AC#1/#4: it reaches the glass as an always-visible line adjacent
    to the control, not only as a mouse tooltip a keyboard user can never reach.
    Painted at both sizes, no hover. AC#2: the reason names the next step
    (Settings ▸ Providers & Models).

    Args:
        size: The terminal dimensions under test.
    """
    host = _host()  # the test config configures no analysis provider
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        await _switch_to_analysis(screen, pilot)
        reason_line = screen.query_one("#library-media-analysis-generate-reason", Static)
        painted = _painted(host, reason_line.region).replace("\n", " ")
        assert "No analysis provider is configured" in painted, painted
        # AC#2: the next step, not just the fault.
        assert "Settings" in painted, painted
        assert "Providers & Models" in painted, painted


@pytest.mark.asyncio
async def test_reader_generate_reason_line_is_absent_when_a_provider_is_ready():
    """task-31981: with a ready provider the inline reason line is gone and
    Generate is live -- the line is the blocker's carrier, not chrome."""
    host = _analysed_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                library_screen_module,
                "analysis_unavailable_reason",
                lambda *_a, **_k: "",
            )
            await _open_first_reader_row(screen, pilot)
            await _switch_to_analysis(screen, pilot)
            assert not screen.query("#library-media-analysis-generate-reason")
            generate = screen.query_one("#library-media-analysis-generate", Button)
            assert generate.disabled is False


@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
@pytest.mark.asyncio
async def test_select_mode_analyze_reason_is_painted_inline_not_hover_only(size):
    """The select-mode bulk Analyze reason paints inline, like Generate's.

    task-31981 AC#1/#4: the select-mode bulk Analyze gates on the same provider
    condition, and its reason must paint inline too -- same silence, same fix,
    at both sizes with no hover.

    Args:
        size: The terminal dimensions under test.
    """
    host = _host()  # the test config configures no analysis provider
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._toggle_library_media_select_mode()
        await _wait_for_selector(screen, pilot, "#library-media-analyze-selected")
        await pilot.pause()
        analyze = screen.query_one("#library-media-analyze-selected", Button)
        assert analyze.disabled is True
        reason_line = screen.query_one("#library-media-analyze-selected-reason", Static)
        painted = _painted(host, reason_line.region).replace("\n", " ")
        assert "No analysis provider is configured" in painted, painted
        assert "Providers & Models" in painted, painted


@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
@pytest.mark.asyncio
async def test_select_mode_bulk_reason_is_painted_with_nothing_selected(size):
    """task-32045 (critique #7 P2): zero selected dims Export/Review/Delete
    with the "○" marker but said nothing inline, while Analyze already
    explains its own block (task-31981). Export/Review/Delete share one
    gate (``selected_count == 0``), so one reason line -- same
    ``.library-media-action-reason`` grammar -- covers all three rather
    than repeating per button.

    Args:
        size: The terminal dimensions under test.
    """
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._toggle_library_media_select_mode()
        await _wait_for_selector(screen, pilot, "#library-media-select-actions")
        await pilot.pause()
        export_btn = screen.query_one("#library-media-export-selected", Button)
        review_btn = screen.query_one("#library-media-review-selected", Button)
        delete_btn = screen.query_one("#library-media-delete-selected", Button)
        assert export_btn.disabled is True
        assert review_btn.disabled is True
        assert delete_btn.disabled is True
        reason_line = screen.query_one("#library-media-select-bulk-reason", Static)
        assert reason_line.styles.visibility == "visible"
        painted = _painted(host, reason_line.region).replace("\n", " ")
        assert "Select items to enable" in painted, painted

        screen.query_one("#library-media-row-0").press()
        await pilot.pause()
        # task-252 Tier 1: a single row press is patched in place (never a
        # recompose), so the line stays mounted but hidden -- "gone" means
        # ``visibility: hidden`` (nothing painted; the row list below keeps
        # its height, unlike ``display=False`` which would shift it).
        hidden_reason = screen.query_one("#library-media-select-bulk-reason", Static)
        assert hidden_reason.styles.visibility == "hidden"
        assert "Select items to enable" not in _painted(
            host, hidden_reason.region
        ).replace("\n", " ")
        assert (
            screen.query_one("#library-media-export-selected", Button).disabled
            is False
        )
        assert (
            screen.query_one("#library-media-review-selected", Button).disabled
            is False
        )
        assert (
            screen.query_one("#library-media-delete-selected", Button).disabled
            is False
        )


@pytest.mark.asyncio
async def test_select_mode_empty_refresh_does_not_ask_to_select_nothing():
    """task-32085 AC#1 (Qodo #7): the zero-selection reason must not paint on a
    SUCCESSFUL empty list.

    task-32045 made ``Select items to enable.`` visible whenever
    ``selected_count == 0``, without checking that any row exists -- so select
    mode surviving a refresh that returns zero rows asked the user to select
    from nothing. The line stays MOUNTED (height reserved, per task-32045's
    visibility-not-display layout fix) but paints nothing when there is no
    selectable row.
    """
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_two_media_items())
    service = app.media_reading_scope_service
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._toggle_library_media_select_mode()
        await _wait_for_selector(screen, pilot, "#library-media-select-actions")
        await pilot.pause()
        # Baseline: with rows present and nothing selected, the reason paints.
        reason = screen.query_one("#library-media-select-bulk-reason", Static)
        assert reason.styles.visibility == "visible"

        # A successful refresh empties the list while select mode survives.
        controller = screen._library_media_browse_controller
        service.media_items = []
        screen._request_library_media_browse(
            controller.mutation_refresh_scope,
            focus_identity=None,
        )
        await _wait_for_condition(
            pilot,
            lambda: controller.applied_result is not None
            and controller.applied_result.total == 0
            and not controller.loading,
            message="Empty Media refresh never applied while in select mode.",
        )
        await pilot.pause()

        assert screen._media_state.select_mode is True
        empty_reason = screen.query("#library-media-select-bulk-reason")
        assert empty_reason, "reason line must stay mounted for layout stability"
        empty_reason_widget = empty_reason.first(Static)
        # No selectable row -> kept mounted via ``visibility: hidden`` (the
        # height-reserving mechanism, NOT ``display: none``), painting nothing.
        assert empty_reason_widget.styles.visibility == "hidden"
        canvas_region = screen.query_one("#library-media-canvas").region
        assert "Select items to enable" not in _painted(
            host, canvas_region
        ).replace("\n", " ")


@pytest.mark.asyncio
async def test_select_mode_analyze_reason_refreshes_on_resume():
    """task-32039 AC#2: a provider configured mid-session clears the gate on return.

    The select-mode bulk-Analyze reason is memoised for the whole select-mode
    session. Configuring a provider while Library is suspended must not leave a
    stale "no provider" memo gating the action -- ``on_screen_resume``
    re-resolves it, so a provider configured mid-session is reflected without a
    restart.
    """
    host = _host()  # the test config configures no analysis provider
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await pilot.press("s")  # enter select mode; arms row focus
        await pilot.pause()
        await pilot.press("space")  # check a row so count is not the gate
        await pilot.pause()
        assert screen._media_state.row_selection.count == 1
        await _wait_for_selector(screen, pilot, "#library-media-analyze-selected")
        analyze = screen.query_one("#library-media-analyze-selected", Button)
        assert analyze.disabled is True
        # The memo is now populated with the no-provider reason.
        assert screen._library_media_analyze_reason(), "reason memo never populated"

        with pytest.MonkeyPatch.context() as mp:
            # A provider is configured mid-session (Library was suspended).
            mp.setattr(
                library_screen_module,
                "analysis_unavailable_reason",
                lambda *_a, **_k: "",
            )
            # Without the resume refresh the stale memo still gates it.
            assert screen._library_media_analyze_reason(), "memo cleared too early"
            screen.on_screen_resume()
            await _wait_for_condition(
                pilot,
                lambda: not screen.query_one(
                    "#library-media-analyze-selected", Button
                ).disabled,
                message="The bulk Analyze gate never cleared after resume.",
            )
            await pilot.pause()
            assert screen._library_media_analyze_reason() == ""
            assert not screen.query("#library-media-analyze-selected-reason")


@pytest.mark.asyncio
async def test_the_analysis_provider_is_resolved_only_on_the_tab_that_shows_it(
    monkeypatch,
):
    """Review I1: ``resolve_ingest_analysis_provider`` can shell out to the
    macOS keychain (Anthropic subscription auth, 5s TTL) -- so resolving it
    on EVERY viewer sync would put a synchronous subprocess on the Textual
    event loop for every Reader gesture. Only ``_compose_analysis`` consumes
    the reason, so only the Analysis tab pays for it."""
    from tldw_chatbook.Library.ingest_analysis import IngestAnalysisResolution
    from tldw_chatbook.UI.Screens import library_screen as library_screen_module

    calls: list[object] = []
    not_ready = IngestAnalysisResolution(
        provider="",
        api_key=None,
        ready=False,
        short_reason="no analysis provider is configured",
        hint="",
    )

    def _recording_resolver(config):
        calls.append(config)
        return not_ready

    monkeypatch.setattr(
        library_screen_module,
        "resolve_ingest_analysis_provider",
        _recording_resolver,
    )
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)

        calls.clear()
        screen._sync_library_media_viewer_or_recompose()
        await pilot.pause()
        assert calls == [], (
            f"the Read tab resolved the analysis provider {len(calls)} time(s)"
        )

        await _switch_to_analysis(screen, pilot)
        calls.clear()
        screen._sync_library_media_viewer_or_recompose()
        await pilot.pause()
        assert len(calls) == 1, (
            f"the Analysis tab resolved the provider {len(calls)} times per sync"
        )
        # And the gate did not cost the feature: the reason still lands.
        generate = screen.query_one("#library-media-analysis-generate", Button)
        assert generate.disabled is True


@pytest.mark.asyncio
async def test_generate_is_live_when_the_configured_provider_is_ready(monkeypatch):
    """The counterpart: a ready resolution leaves the action untouched --
    no marker, no tooltip, no disabled state."""
    from tldw_chatbook.Library.ingest_analysis import IngestAnalysisResolution
    from tldw_chatbook.UI.Screens import library_screen as library_screen_module

    monkeypatch.setattr(
        library_screen_module,
        "resolve_ingest_analysis_provider",
        lambda _config: IngestAnalysisResolution(
            provider="OpenAI",
            api_key="sk-test",
            ready=True,
            short_reason="",
            hint="",
            dispatch_name="openai",
        ),
    )
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        await _switch_to_analysis(screen, pilot)
        generate = screen.query_one("#library-media-analysis-generate", Button)
        assert generate.disabled is False
        assert str(generate.label) == "Generate"
        assert generate.tooltip is None


# ---------------------------------------------------------------------------
# task-31631: select mode is reachable by keyboard, and "Done" leaves the
# "sort:" slot. Critique #5 P1: from the rail, "s" entered select mode but
# F6, Down and Space all no-opped with no focus painted anywhere -- only a
# mouse click on the one-cell "☐" seeded focus. Painted on purpose (the
# task-31221 lesson): the row's focus cue is a STYLE (focus background +
# bold underline, ``outline: none``), so a region assertion cannot tell a
# focused row from an unfocused one.
# ---------------------------------------------------------------------------


async def _enter_media_select_mode(screen, pilot):
    """Stand on the rail row, press "s", and wait for select mode to settle."""
    screen.set_focus(screen.query_one("#library-row-browse-media", Button))
    await pilot.pause()
    await pilot.press("s")
    await _wait_for_condition(
        pilot,
        lambda: screen._media_state.select_mode,
        message="Select mode never engaged after 's'.",
    )
    await pilot.pause()
    await pilot.pause()


@pytest.mark.parametrize("size", [(235, 52), (100, 30)])
@pytest.mark.asyncio
async def test_select_mode_entry_focuses_a_row_so_down_and_space_work(size):
    """task-31631 AC#1/AC#4: "s" from the rail lands a painted focus cue on a
    ROW -- and keeps it there when a background worker's recompose rebuilds
    the rows -- so Down and Space work immediately, as the footer promises."""
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _enter_media_select_mode(screen, pilot)

        focused = screen.focused
        assert focused is not None, "nothing is focused after entering select mode"
        assert focused.has_class("library-media-row"), (
            f"focus landed on {focused!r}, not a media row"
        )
        assert _row_is_painted_focused(host, focused), _painted(host, focused.region)
        # The cue is the row's own, not something every row paints.
        other = screen.query_one("#library-media-row-1", Button)
        assert not any(
            style.underline for _text, style in _painted_cells(host, other.region)
        ), _painted(host, other.region)

        # Any of the screen's background workers ends in its own whole-screen
        # recompose, which rebuilds the row Buttons. A one-shot focus dies
        # with the widget it was set on; the armed entry-focus request does
        # not. (The Reader's general case is task-31567's.)
        screen.refresh(recompose=True)
        await pilot.pause()
        await pilot.pause()
        await pilot.pause()
        focused_after = screen.focused
        assert (
            focused_after is not None
            and focused_after.has_class("library-media-row")
            and _row_is_painted_focused(host, focused_after)
        ), f"focus was {focused_after!r} after a background recompose"

        # ...and the keys the footer promises work immediately.
        await pilot.press("down")
        await pilot.press("space")
        await _wait_for_condition(
            pilot,
            lambda: screen._media_state.row_selection.count == 1,
            message="Down then Space did not select a row.",
        )
        await pilot.pause()
        canvas = screen.query_one("#library-media-canvas")
        assert "1 selected" in _painted(host, canvas.region)


_BROWSE_TOOLBAR_SLOTS = (
    "#library-media-type-filter",
    "#library-media-sort",
    "#library-media-export",
    "#library-media-trash-open",
    "#library-media-review",
)


@pytest.mark.parametrize("size", [(235, 52), (100, 30)])
@pytest.mark.asyncio
async def test_done_does_not_take_the_sort_slot(size):
    """task-31631 AC#3: "Done" must not land where "sort:" sat in browse mode.

    The habitual click on the sort chooser otherwise silently becomes "leave
    select mode and discard the selection" -- so the pin covers every browse
    toolbar slot, not only sort's: trading sort's cells for Export…'s would
    be the same bug wearing a different label.
    """
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        browse_slots = {}
        for selector in _BROWSE_TOOLBAR_SLOTS:
            region = screen.query_one(selector, Button).region
            assert region.width > 0, selector
            browse_slots[selector] = region

        await _enter_media_select_mode(screen, pilot)
        done = screen.query_one("#library-media-select-toggle", Button)
        assert str(done.label) == "Done"
        assert done.region.width > 0
        for selector, region in browse_slots.items():
            overlaps = not (
                done.region.right <= region.x
                or done.region.x >= region.right
                or done.region.bottom <= region.y
                or done.region.y >= region.bottom
            )
            assert not overlaps, (
                f"Done at {done.region} overlaps the browse-mode {selector} "
                f"slot {region}"
            )
        # Still a whole word, painted inside the pane.
        canvas = screen.query_one("#library-media-canvas")
        assert "Done" in _painted(host, canvas.region)
        assert done.region.right <= canvas.region.right


# ---------------------------------------------------------------------------
# task-31634: the Reader pane's focus must not be colour-only. Critique #5
# measured its top border recolouring 1.01:1 -> 6.96:1 with BYTE-IDENTICAL
# glyphs, so a monochrome or colour-blind reader gets no signal that F6
# landed. Buttons on the same screen already change glyphs (heavy outline).
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_reader_focus_changes_border_glyphs_not_only_colour():
    """task-31634 AC#1/AC#2: the focused Reader box paints a HEAVY border.

    Plain text on purpose (the AC is "visible in a plain-text capture"):
    the previous cue was a recolour of the same ``─`` glyphs, which a
    plain capture cannot tell apart at all.
    """
    host = _global_key_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        box = screen.query_one("#library-media-viewer-content")
        screen.query_one("#library-search-input", Input).focus()
        await pilot.pause()
        unfocused = _top_border_row(host, box)

        for _ in range(6):
            await pilot.press("f6")
            await pilot.pause()
            if str(getattr(screen.focused, "id", "")) in {
                "library-media-viewer-content-text",
                "library-media-viewer-content",
            }:
                break
        assert str(getattr(screen.focused, "id", "")) in {
            "library-media-viewer-content-text",
            "library-media-viewer-content",
        }, screen.focused
        focused = _top_border_row(host, screen.query_one(
            "#library-media-viewer-content"
        ))

        assert focused != unfocused, (
            f"focus is colour-only: {unfocused!r} == {focused!r}"
        )
        assert unfocused.startswith("┌") and "─" in unfocused, unfocused
        assert focused.startswith("┏") and "━" in focused, focused
        assert "─" not in focused, focused


# --- task-31633 AC#3: More is one row, not a push -------------------------
#
# Critique #5 P1 (capture 10): the Reader's "More" disclosure composed a
# bare ``Vertical`` above the mode row. An unstyled Vertical defaults to
# ``1fr``, so it claimed 19 rows for three one-row buttons -- pushing the
# tab row and the whole reading body down and leaving ~16 painted-blank
# rows before the content resumed. These pin the row, not the widget: the
# painted tab-row offset at both the wide and the compact size.

_MORE_ACTION_LABELS = (
    "Edit metadata",
    "Open original",
    "Open manager",
    "Move to trash",
)


def _four_action_host() -> LibraryProductionCSSHarness:
    """A Reader whose items carry a URL, so More renders all four actions.

    ``Open original`` is composed only when the item has an original
    source, and the shared fixture has none -- without a URL the row this
    test measures would be three buttons wide and the "all four readable"
    assertion would be vacuous.
    """
    app = _build_media_test_app()
    items = [
        {**item, "url": f"https://example.test/{item['id']}"}
        for item in _two_media_items()
    ]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


async def _open_reader_more(screen, pilot):
    screen.query_one("#library-media-reader-more", Button).press()
    actions = await _wait_for_selector(
        screen, pilot, "#library-media-reader-more-actions"
    )
    # The disclosure's row has to be laid out before a caller can measure it;
    # its own region is the thing that settles, so wait on that rather than on
    # a fixed number of frames.
    await _wait_for_condition(
        pilot,
        lambda: actions.region.height > 0,
        message="The More actions row never took a painted region.",
    )
    return actions


def _reader_row_tops(screen) -> tuple[int, int]:
    """The painted top row of the tab strip and of the reading body."""
    return (
        screen.query_one("#library-media-reader-mode-toolbar").region.y,
        screen.query_one("#library-media-reader-mode-read").region.y,
    )


@pytest.mark.asyncio
async def test_more_opens_one_row_and_moves_the_reader_body_by_one():
    """At 235x52 More costs exactly one row, and all four actions paint on it."""
    host = _four_action_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        closed_tabs, closed_body = _reader_row_tops(screen)

        actions = await _open_reader_more(screen, pilot)
        open_tabs, open_body = _reader_row_tops(screen)

        assert actions.region.height == 1, actions.region
        assert open_tabs - closed_tabs == 1, (closed_tabs, open_tabs)
        assert open_body - closed_body == 1, (closed_body, open_body)

        painted = _painted(host, actions.region)
        for label in _MORE_ACTION_LABELS:
            assert label in painted, painted


@pytest.mark.asyncio
async def test_more_reads_as_an_open_disclosure_while_it_is_open():
    """The primary row paints "More ▴" while open and "More" when closed."""
    host = _four_action_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        primary = screen.query_one("#library-media-reader-primary-toolbar")
        assert "More ▴" not in _painted(host, primary.region)

        await _open_reader_more(screen, pilot)
        primary = screen.query_one("#library-media-reader-primary-toolbar")
        assert "More ▴" in _painted(host, primary.region)

        screen.query_one("#library-media-reader-more", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query("#library-media-reader-more-actions"),
            message="More never closed.",
        )
        primary = screen.query_one("#library-media-reader-primary-toolbar")
        assert "More ▴" not in _painted(host, primary.region)


@pytest.mark.asyncio
async def test_more_toggle_leaves_focus_on_the_more_button():
    """Toggling the disclosure never hands focus to the row it opened.

    The Reader recomposes on this toggle, so the focused identity is
    whatever the restore seam last saw (the Items row that opened the
    Reader). The disclosure owns its own focus target explicitly.
    """
    host = _four_action_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)

        await _open_reader_more(screen, pilot)
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None)
            == "library-media-reader-more",
            message=lambda: f"Opening More never focused it: {screen.focused!r}.",
        )

        screen.query_one("#library-media-reader-more", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: not screen.query("#library-media-reader-more-actions"),
            message="More never closed.",
        )
        await _wait_for_condition(
            pilot,
            lambda: getattr(screen.focused, "id", None)
            == "library-media-reader-more",
            message=lambda: f"Closing More never focused it: {screen.focused!r}.",
        )


@pytest.mark.asyncio
async def test_more_stays_compact_at_the_narrow_reader_width():
    """At 100x30 the four actions fit or wrap once; the body moves <= 2 rows."""
    host = _four_action_host()
    async with host.run_test(size=(100, 30)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        closed_tabs, closed_body = _reader_row_tops(screen)

        actions = await _open_reader_more(screen, pilot)
        open_tabs, open_body = _reader_row_tops(screen)

        assert actions.region.height <= 2, actions.region
        assert 1 <= open_tabs - closed_tabs <= 2, (closed_tabs, open_tabs)
        assert 1 <= open_body - closed_body <= 2, (closed_body, open_body)

        painted = _painted(host, actions.region)
        for label in _MORE_ACTION_LABELS:
            assert label in painted, painted


async def _force_media_page_failure(host, screen, pilot, exc: BaseException):
    """Fail the applied Media page in place and return its call counter.

    A page-1 request that fails AFTER page 1 applied is the critique's
    exact state ("Couldn't load page 1."): the rows stay retained, the
    pager keeps its counters, and the only recovery is Retry.
    """
    calls: list[int] = []

    async def _fails(**_kwargs):
        calls.append(1)
        raise exc

    host.app_instance.media_reading_scope_service.search_media = _fails
    controller = screen._library_media_browse_controller
    screen._request_library_media_page(1, focus_identity=None)
    await _wait_for_condition(
        pilot,
        lambda: (
            controller.failure is not None
            and not controller.loading
            # The callout composes before the row scroll: settle on BOTH the
            # mounted callout and the remounted retained rows, not on the
            # controller alone (a single pause raced the mount elsewhere).
            and bool(screen.query("#library-media-load-failure-copy"))
            and len(screen.query(".library-media-row"))
            == len(controller.retained_items)
        ),
        message="The forced Media page failure never settled.",
    )
    await pilot.pause()
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
async def test_media_load_failure_paints_one_callout_with_retry_inside(size):
    """task-31632 AC#1: the reason and its Retry, painted together.

    Critique #5 P1: ``Couldn't load page 1.`` painted as a bare sentence
    with no reason, and its only Retry 34 rows below in the pager strip.
    """
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _force_media_page_failure(
            host, screen, pilot, sqlite3.OperationalError("database is locked")
        )

        callout = screen.query_one("#library-media-load-failure")
        copy = screen.query_one("#library-media-load-failure-copy", Static)
        retry = screen.query_one("#library-media-retry", Button)

        painted = " ".join(_painted(host, copy.region).split())
        assert painted == "Couldn't load page 1 · database is locked", painted
        assert "Retry" in _painted(host, retry.region)

        # The Retry is IN the callout, on the message's own row or the one
        # directly below it -- never the pager strip 34 rows down.
        assert retry in callout.query(Button)
        assert 0 <= retry.region.y - copy.region.y <= 1, (
            copy.region,
            retry.region,
        )
        assert retry.region.y - copy.region.y <= 3

        # Exactly one Retry, and the pager strip does not carry a second.
        assert len(screen.query("#library-media-retry")) == 1
        pager = screen.query_one("#library-media-pager")
        assert not pager.query("#library-media-retry")

        # The retained rows stay exactly as they were.
        assert len(screen.query(".library-media-row")) == 2


@pytest.mark.asyncio
async def test_media_load_failure_callout_retry_issues_a_new_request():
    """task-31632 AC#1: the Retry inside the callout is the live one."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller
        calls = await _force_media_page_failure(
            host, screen, pilot, sqlite3.OperationalError("database is locked")
        )
        assert len(calls) == 1

        screen.query_one("#library-media-load-failure").query_one(
            "#library-media-retry", Button
        ).press()
        await _wait_for_condition(
            pilot,
            lambda: len(calls) == 2,
            message="The callout's Retry never issued a new request.",
        )
        await _wait_for_condition(
            pilot,
            lambda: not controller.loading,
            message="The retried request never settled.",
        )
        await pilot.pause()

        # Still one callout, repainted with the fresh reason -- never a
        # silent press.
        assert len(screen.query("#library-media-load-failure")) == 1
        assert controller.failure is not None


@pytest.mark.asyncio
async def test_media_facet_failure_paints_the_same_callout():
    """task-31632 AC#1 (Task 2 review carry-over): the type-list failure was
    only ever covered at controller level. It reaches the SAME callout, with
    the same single Retry, because the type list has no control of its own.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller

        async def _fails(**_kwargs):
            raise sqlite3.OperationalError("database is locked")

        host.app_instance.media_reading_scope_service.list_library_media_types = _fails
        screen._request_library_media_facets()
        await _wait_for_condition(
            pilot,
            lambda: controller.facet_failure is not None and not controller.facet_loading,
            message="The forced Media facet failure never settled.",
        )
        await pilot.pause()

        assert controller.page_failure is None
        callout = screen.query_one("#library-media-load-failure")
        copy = screen.query_one("#library-media-load-failure-copy", Static)
        painted = " ".join(_painted(host, copy.region).split())
        assert painted == "Couldn't load media types · database is locked", painted
        retry = screen.query_one("#library-media-retry", Button)
        assert retry in callout.query(Button)
        assert len(screen.query("#library-media-retry")) == 1


@pytest.mark.asyncio
async def test_repeated_media_load_failure_names_the_reopen_recovery():
    """task-31982 AC#2: a second consecutive failed Retry stops repeating.

    The critique's residue: the callout's Retry re-issued on the same failed
    path and repainted the identical sentence with no next step. When the
    same reason recurs on a consecutive Retry the message must name the
    recovery action instead -- reopen Chatbook to reconnect to the store.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller
        calls = await _force_media_page_failure(
            host, screen, pilot, sqlite3.OperationalError("database is locked")
        )

        copy = screen.query_one("#library-media-load-failure-copy", Static)
        first = " ".join(_painted(host, copy.region).split())
        # The FIRST failure is silent about the recovery step -- one honest
        # sentence, exactly as before.
        assert first == "Couldn't load page 1 · database is locked", first

        screen.query_one("#library-media-load-failure").query_one(
            "#library-media-retry", Button
        ).press()
        await _wait_for_condition(
            pilot,
            lambda: len(calls) == 2,
            message="The callout's Retry never re-issued.",
        )
        await _wait_for_condition(
            pilot,
            lambda: not controller.loading,
            message="The retried request never settled.",
        )
        await pilot.pause()

        copy = screen.query_one("#library-media-load-failure-copy", Static)
        second = " ".join(_painted(host, copy.region).split())
        assert second.startswith("Couldn't load page 1 · database is locked"), second
        assert "reopen Chatbook to reconnect to the media database" in second, second


@pytest.mark.asyncio
async def test_changed_page_context_failure_is_not_a_repeated_retry():
    """task-32039 AC#1: a same-reason failure in a NEW context is not a Retry.

    The repeated-fault episode is scoped to its request context. A page (or
    query/type) change that happens to hit the same reason is a first failure
    of that context, so it must NOT wear the reopen recovery step -- that is
    reserved for a genuine consecutive Retry of the same context.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller
        calls = await _force_media_page_failure(
            host, screen, pilot, sqlite3.OperationalError("database is locked")
        )

        copy = screen.query_one("#library-media-load-failure-copy", Static)
        first = " ".join(_painted(host, copy.region).split())
        assert first == "Couldn't load page 1 · database is locked", first

        # A DIFFERENT page hits the same reason -- a changed context, never a
        # consecutive Retry of the one that just failed.
        screen._request_library_media_page(2, focus_identity=None)
        await _wait_for_condition(
            pilot,
            lambda: len(calls) >= 2,
            message="The page-2 request never issued.",
        )
        await _wait_for_condition(
            pilot,
            lambda: not controller.loading,
            message="The page-2 request never settled.",
        )
        await pilot.pause()

        copy = screen.query_one("#library-media-load-failure-copy", Static)
        second = " ".join(_painted(host, copy.region).split())
        assert second.startswith("Couldn't load page 2 · database is locked"), second
        assert "reopen Chatbook to reconnect to the media database" not in second, second


@pytest.mark.asyncio
async def test_resume_refresh_failure_is_not_a_repeated_retry():
    """task-32039 AC#1: a Library resume auto-refresh is a new visit, not a Retry.

    Leaving and returning re-issues the same scope through ``on_screen_resume``.
    That failing with the same reason is the FIRST failure of a new visit, so
    the reopen recovery step must not appear -- only a genuine consecutive
    Retry within a visit escalates.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller
        calls = await _force_media_page_failure(
            host, screen, pilot, sqlite3.OperationalError("database is locked")
        )

        copy = screen.query_one("#library-media-load-failure-copy", Static)
        first = " ".join(_painted(host, copy.region).split())
        assert "reopen Chatbook to reconnect to the media database" not in first, first

        # Leave and return: the resume auto-refresh re-issues the same page and
        # fails with the same reason. A new visit, not a Retry.
        screen.on_screen_resume()
        await _wait_for_condition(
            pilot,
            lambda: len(calls) >= 2,
            message="The resume refresh never re-issued the page request.",
        )
        await _wait_for_condition(
            pilot,
            lambda: not controller.loading,
            message="The resume refresh never settled.",
        )
        await pilot.pause()

        copy = screen.query_one("#library-media-load-failure-copy", Static)
        second = " ".join(_painted(host, copy.region).split())
        assert second.startswith("Couldn't load page 1 · database is locked"), second
        assert "reopen Chatbook to reconnect to the media database" not in second, second


@pytest.mark.asyncio
async def test_facet_only_failure_leaves_the_row_supported_actions_live():
    """task-31982 AC#1/#4: the facet read and the row read are independent.

    ``list_library_media_types`` and ``search_media`` are separate queries on
    separate worker groups with independent failure fences, so a facet
    failure over a healthy page must NOT disable the actions the row data
    supports -- Export, Review, Select and Trash stay live because their own
    read succeeded.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        controller = screen._library_media_browse_controller

        async def _fails(**_kwargs):
            raise sqlite3.OperationalError("database is locked")

        host.app_instance.media_reading_scope_service.list_library_media_types = _fails
        screen._request_library_media_facets()
        await _wait_for_condition(
            pilot,
            lambda: (
                controller.facet_failure is not None
                and not controller.facet_loading
            ),
            message="The forced Media facet failure never settled.",
        )
        await pilot.pause()

        # The page read succeeded, its rows are retained, and the "nothing to
        # select" predicate the whole-list gate reads stays False.
        assert controller.page_failure is None
        assert len(controller.retained_items) == 2
        assert not screen._library_media_list_unselectable()

        # ...so every action the row data supports stays live and un-gated.
        for widget_id, label in (
            ("#library-media-export", "Export…"),
            ("#library-media-review", "Review these"),
            ("#library-media-trash-open", "Trash"),
            ("#library-media-select-toggle", "Select"),
        ):
            button = screen.query_one(widget_id, Button)
            assert not button.disabled, widget_id
            assert str(button.label) == label, widget_id


@pytest.mark.asyncio
async def test_media_failure_callout_tint_follows_the_severity():
    """task-31632 AC#1: a timeout and a hard failure do not paint alike."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _force_media_page_failure(host, screen, pilot, TimeoutError())

        callout = screen.query_one("#library-media-load-failure")
        copy = screen.query_one("#library-media-load-failure-copy", Static)
        assert " ".join(_painted(host, copy.region).split()) == (
            "Couldn't load page 1 · timed out"
        )
        assert not callout.has_class("is-blocked")
        timeout_border = callout.styles.border_top

        await _force_media_page_failure(host, screen, pilot, RuntimeError("boom"))
        callout = screen.query_one("#library-media-load-failure")
        assert callout.has_class("is-blocked")
        assert callout.styles.border_top != timeout_border, timeout_border


# ---------------------------------------------------------------------------
# task-31633 (critique #5 P1): two rows per item, not three.
#
# Painted, not region-only: the third row was a bottom margin on the row
# button, so every region assertion on the button itself already read "2" --
# only the painted list shows the blank row that margin bought, and the eleven
# items it cost a 52-row terminal.
# ---------------------------------------------------------------------------

_ROWS_PER_MEDIA_ITEM = 2


def _fifteen_item_host() -> LibraryProductionCSSHarness:
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_many_media_items(15))
    return LibraryProductionCSSHarness(app)


def _blank_row_focus_bar(line: str) -> str:
    """Blank the focused row's ``border-left: thick`` cell (task-32199).

    Entering the Media list parks DOM focus on the first row (task-2856),
    and since task-31983 a focused row paints a ``█`` bar in the cell its
    left padding otherwise occupies (``.library-media-row:focus``). That
    bar is chrome, not row text, and ONLY the focused row carries it -- left
    in the capture it makes row 0 incomparable with its own siblings, which
    is what these row-text assertions compare. One cell in, one cell out, so
    every column index in this file still lines up.
    """
    return f" {line[1:]}" if line.startswith("█") else line


def _painted_item_lines(host, screen) -> list[str]:
    """Return the painted row-scroll lines of the Media list."""
    scroll = screen.query_one("#library-media-row-scroll")
    strips = list(host.screen._compositor.render_strips())
    return [
        _blank_row_focus_bar(
            strips[y].crop(scroll.region.x, scroll.region.right).text
        )
        for y in range(scroll.region.y, min(scroll.region.bottom, len(strips)))
    ]


@pytest.mark.asyncio
async def test_media_items_paint_two_rows_each_with_no_blank_row_between():
    """Fifteen seeded items all paint in a 52-row terminal, two rows each."""
    host = _fifteen_item_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        for _ in range(4):
            await pilot.pause()

        lines = _painted_item_lines(host, screen)
        titles = [
            index
            for index, line in enumerate(lines)
            if "Media item " in line
        ]

        assert len(titles) >= 15, (len(titles), lines)
        for previous, current in zip(titles, titles[1:]):
            assert current - previous == _ROWS_PER_MEDIA_ITEM, (titles, lines)
            meta = lines[previous + 1].strip()
            assert meta.split(" ", 1)[0] in {"video", "audio", "PDF"}, (
                meta,
                lines,
            )


# ---------------------------------------------------------------------------
# task-28008 / task-28009 (render half): the row's state slot and the
# "analysed" segment of its secondary line.
# ---------------------------------------------------------------------------


_ANALYSED_SECONDARY = "document · updated 5m · analysed"


def _review_state_items(count: int = 4, analysed: int = 2) -> list[dict]:
    """``count`` ``document`` items, the first ``analysed`` of them analysed.

    The stamp is fixed 5m30s back so every row's age label is "5m" and the
    secondary line is exactly the 30-cell ``document · updated 5m · analysed`` the
    Items pane has to hold.
    """
    stamp = (
        datetime.now(timezone.utc) - timedelta(minutes=5, seconds=30)
    ).isoformat()
    return [
        {
            "id": f"media-{index}",
            "title": f"Doc {index}",
            "type": "document",
            "last_modified": stamp,
            "content": f"Body of doc {index}.",
            "version": 1,
            "has_analysis": index <= analysed,
        }
        for index in range(1, count + 1)
    ]


def _review_state_host(count: int = 4, analysed: int = 2):
    """A production-CSS Library host over :func:`_review_state_items`."""
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_review_state_items(count, analysed))
    return LibraryProductionCSSHarness(app)


def _painted_media_rows(host, screen) -> tuple[list[str], list[str]]:
    """Return the painted (title lines, secondary lines) of the Media list."""
    lines = _painted_item_lines(host, screen)
    titles = [line for line in lines if "Doc " in line]
    secondaries = [line for line in lines if "document · " in line]
    return titles, secondaries


@pytest.mark.parametrize(
    ("size", "items_width"),
    [((235, 52), 132), ((100, 30), 44)],
    ids=["wide", "narrow"],
)
@pytest.mark.asyncio
async def test_media_rows_paint_analysed_only_for_analysed_items(size, items_width):
    """task-28008: the row says which items already carry an analysis, in words.

    What this pins about width: each parametrized size resolves the Items
    pane to its own AUTOMATIC width, and the 30-cell
    ``document · updated 5m · analysed`` paints whole in it, indented, with room to
    spare. The narrower 36-cell FLOOR is pinned separately by
    ``test_analysed_secondary_is_bounded_at_custom_items_widths`` -- a
    ``>= 36`` assertion here would have claimed a floor that never ran,
    which is why the exact automatic width is asserted instead.

    task-32199: both numbers used to be 52 and had been stale on dev since
    the resolver changed under them -- 9187bc0307 (task-31979) hands the
    empty Reader's unused width to the Items list at the wide size, and the
    below-64 Media stage work (c668aaec5e / db86a39b19) re-fitted the narrow
    one. Both new widths are still far above the 30-cell secondary, so what
    the assertion means is unchanged; the resolver's own contract lives in
    Tests/UI/test_library_adaptive_reader_shell.py.

    Args:
        size: Terminal dimensions ``(columns, rows)`` the harness runs at.
        items_width: The Items pane's automatically resolved width at that
            terminal size -- the exact number, so a resolver change lands
            here instead of hiding behind a ``>=`` floor.
    """
    host = _review_state_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        for _ in range(3):
            await pilot.pause()

        _titles, secondaries = _painted_media_rows(host, screen)
        assert len(secondaries) == 4, secondaries
        assert [line.strip() for line in secondaries[:2]] == [
            _ANALYSED_SECONDARY,
            _ANALYSED_SECONDARY,
        ], secondaries
        assert [line.strip() for line in secondaries[2:]] == [
            "document · updated 5m",
            "document · updated 5m",
        ], secondaries
        assert _items_pane_width(screen) == items_width


@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
@pytest.mark.asyncio
async def test_media_rows_paint_the_active_sets_review_state(size):
    """task-28009: every row in the active set carries a state glyph -- `·`
    until it is reviewed, `✓` after -- and rows outside it carry neither."""
    host = _review_state_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        service = screen._review_set_service()
        set_id = service.create_review_set(
            "These", origin="browse", items=[(1, "Doc 1"), (2, "Doc 2")]
        )
        _sync_library_canvas(screen, "media")
        for _ in range(3):
            await pilot.pause()

        titles, _secondaries = _painted_media_rows(host, screen)
        assert [line[1] for line in titles] == ["·", "·", " ", " "], titles

        service.mark_item_done(set_id, backing_media_id=1, done=True)
        _sync_library_canvas(screen, "media")
        for _ in range(3):
            await pilot.pause()

        titles, _secondaries = _painted_media_rows(host, screen)
        assert [line[1] for line in titles] == ["✓", "·", " ", " "], titles


@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
@pytest.mark.asyncio
async def test_select_mode_checkbox_replaces_the_review_state_slot(size):
    """Controller ruling (4): one slot. In select mode it is the ☑/☐."""
    host = _review_state_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        service = screen._review_set_service()
        service.create_review_set(
            "These", origin="browse", items=[(1, "Doc 1"), (2, "Doc 2")]
        )
        _sync_library_canvas(screen, "media")
        await pilot.pause()
        await _enter_media_select_mode(screen, pilot)

        titles, _secondaries = _painted_media_rows(host, screen)
        assert [line[1] for line in titles] == ["☐", "☐", "☐", "☐"], titles
        assert "·" not in "".join(line[1] for line in titles), titles


def _review_reader_host(count: int = 2):
    """The same seeded rows, behind a gated detail service so the Reader can
    be settled on a known row before the mark gestures are pressed."""
    items = _review_state_items(count, analysed=1)
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=items)
    service = ControlledDetailMediaService(items)
    app.media_reading_scope_service = service
    return LibraryProductionCSSHarness(app), service


def _painted_slots(host, screen) -> list[str]:
    """The painted one-cell state slot of each Media row, in order."""
    titles, _secondaries = _painted_media_rows(host, screen)
    return [line[1] for line in titles]


@pytest.mark.asyncio
async def test_marking_reviewed_in_the_reader_repaints_the_row_slot():
    """task-28009: `m` and the final `]` change the MARK, not the loaded item.

    Both land on the viewer-scoped sync seam without loading anything, and
    the Items list stays mounted beside the Reader -- so unless that seam
    repaints the rows, the slot stays a gesture behind the banner that just
    moved. Painted on the real screen, because the row markers are exactly
    what a state-only assertion would miss.
    """
    host, service = _review_reader_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        review = screen._review_set_service()
        review.create_review_set(
            "These", origin="browse", items=[(1, "Doc 1"), (2, "Doc 2")]
        )
        _sync_library_canvas(screen, "media")
        await pilot.pause()
        await _load_row_0(screen, service, pilot)
        await _wait_for_condition(
            pilot,
            lambda: _painted_slots(host, screen) == ["·", "·"],
            message="The active set never painted its unreviewed rows.",
        )

        await pilot.press("m")
        await _wait_for_condition(
            pilot,
            lambda: _painted_slots(host, screen) == ["✓", "·"],
            message="`m` did not repaint the loaded row's slot as reviewed.",
        )

        # ...and it is a live read, not a one-shot: un-marking flips it back.
        await pilot.press("m")
        await _wait_for_condition(
            pilot,
            lambda: _painted_slots(host, screen) == ["·", "·"],
            message="A second `m` did not repaint the row as unreviewed.",
        )

        # Advancing marks the row it leaves (that path loads an item and
        # repaints through the selection seam)...
        await _walk_next(screen, service, pilot, expected_row=1)
        await _wait_for_condition(
            pilot,
            lambda: _painted_slots(host, screen) == ["✓", "·"],
            message="] did not repaint the row it left as reviewed.",
        )

        # ...and the final ] on the last item is the completion gesture: it
        # marks in place, loading nothing, so only this seam can repaint it.
        await pilot.press("right_square_bracket")
        await _wait_for_condition(
            pilot,
            lambda: _painted_slots(host, screen) == ["✓", "✓"],
            message="The final ] did not repaint the last row as reviewed.",
        )


# --- PR H2 (Qodo on #2470): every viewer-sync follow-up rides the VIEWER ---
#
# ``_sync_library_media_viewer_or_recompose`` rebuilds the Reader on the
# VIEWER's message pump. A ``screen.call_after_refresh`` follow-up has no
# ordering against that pump: it focuses the control the recompose is about
# to detach, Textual re-picks focus for the pruned widget, and task-31567's
# restore -- whose captured identity went with the same children -- takes
# its list-entry fallback. Measured on 43b0a7440: Escape from inside the
# open More disclosure left focus on ``#library-media-row-0``, outside the
# Reader, so the next Escape acted on the LIST. PR H fixed the More BUTTON;
# these pin the Escape paths, which took the same shape.


async def _open_reader_find(screen, pilot):
    """Open the Reader's Find bar and wait for its input to take focus."""
    screen.query_one("#library-media-reader-find", Button).press()
    search_input = await _wait_for_selector(
        screen, pilot, "#library-media-content-search"
    )
    await _wait_for_condition(
        pilot,
        lambda: search_input.has_focus,
        message=lambda: f"Find never focused its input: {screen.focused!r}.",
    )
    return search_input


def _focus_report(screen) -> str:
    focused = screen.focused
    return (
        f"{focused!r} (id={getattr(focused, 'id', None)!r}, "
        f"attached={getattr(focused, 'is_attached', None)!r})"
    )


async def _settle_focus_on(screen, pilot, control_id: str, what: str) -> None:
    """Wait until ``control_id`` holds focus as a MOUNTED widget."""
    await _wait_for_condition(
        pilot,
        lambda: (
            screen.focused is not None
            and screen.focused.id == control_id
            and screen.focused.is_attached
        ),
        message=lambda: f"{what} left focus on {_focus_report(screen)}.",
    )
    # The identity check the orphan cannot pass: the focused widget IS the
    # one a fresh query returns, not a detached same-id predecessor.
    assert screen.focused is screen.query_one(f"#{control_id}", Button)
    assert screen.focused.is_attached


@pytest.mark.asyncio
async def test_escape_closing_more_lands_on_the_live_more_button():
    """Escape closes the disclosure and leaves focus on the NEW More button.

    Qodo High on #2470 ("Readers lose keys after escape"): the follow-up ran
    on the screen's pump while the viewer rebuilt on its own, so it never
    held the control it named. The first half passes on 43b0a7440 by
    coincidence -- the identity task-31567 captured IS the More button, so
    its restore lands where the follow-up wanted; the second half, from
    inside the disclosure, is where the two differ and the base fails.
    Both then press Enter on whatever holds focus: only a live, mounted More
    button re-opens the row.
    """
    host = _four_action_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        await _open_reader_more(screen, pilot)
        await _settle_focus_on(
            screen, pilot, "library-media-reader-more", "Opening More"
        )

        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: not screen.query("#library-media-reader-more-actions"),
            message="Escape never closed More.",
        )
        await _settle_focus_on(
            screen, pilot, "library-media-reader-more", "Escape closing More"
        )

        # Again from INSIDE the disclosure, where the restore cannot mask the
        # bug: the focused action is recomposed away, so PR F's captured
        # identity is gone and its fallback leaves the Reader entirely (on
        # 43b0a7440 this landed on the media list row). The disclosure's own
        # target has to survive the rebuild.
        await _open_reader_more(screen, pilot)
        screen.query_one("#library-media-edit", Button).focus()
        await pilot.pause()
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: not screen.query("#library-media-reader-more-actions"),
            message="Escape never closed More from inside it.",
        )
        await _settle_focus_on(
            screen,
            pilot,
            "library-media-reader-more",
            "Escape closing More from inside it",
        )

        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-media-reader-more-actions")


@pytest.mark.asyncio
async def test_escape_closing_find_lands_on_the_live_find_button():
    """Both Escape-closes-Find branches land on the mounted Find button.

    First from INSIDE the bar (the branch that reads the focused widget's
    ancestry), then with focus moved out to the content body (the branch
    that consumes an open bar regardless of where focus sits).
    """
    host = _four_action_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)

        await _open_reader_find(screen, pilot)
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: not screen.query("#library-media-content-search-controls"),
            message="Escape never closed the Find bar.",
        )
        await _settle_focus_on(
            screen, pilot, "library-media-reader-find", "Escape closing Find"
        )

        await _open_reader_find(screen, pilot)
        screen.query_one("#library-media-viewer-content").focus()
        await pilot.pause()
        await pilot.press("escape")
        await _wait_for_condition(
            pilot,
            lambda: not screen.query("#library-media-content-search-controls"),
            message="Escape never closed the Find bar from the content body.",
        )
        await _settle_focus_on(
            screen,
            pilot,
            "library-media-reader-find",
            "Escape closing Find from the content body",
        )

        # Keys still work afterwards: Enter on the restored button re-opens.
        await pilot.press("enter")
        await _wait_for_selector(
            screen, pilot, "#library-media-content-search-controls"
        )


# --- PR L (task-31950 / task-31954): the LAST screen-pump follow-ups ---
#
# Two siblings of the Escape defect above survived PR H2, both queued from
# ``_sync_library_media_viewer_state``'s own tail: the edit-Save mutation
# gate and the reading-progress restore. Neither is a focus move, so the
# race shows up as a LOST follow-up rather than a stranded focus -- the
# screen callback flushes while the viewer's new children are not mounted,
# ``query_one`` raises ``NoMatches``, and the gate silently no-ops.


_VIEWER_RECOMPOSE_ANCHOR = re.compile(
    r"_sync_library_media_viewer_or_recompose\(\)|viewer\.refresh\(recompose=True\)"
)


def _screen_pump_follow_ups_after_a_viewer_recompose(
    window: int = 14,
) -> list[tuple[int, str]]:
    """The task-31950 census: a viewer recompose, then ``screen.call_after_refresh``.

    The brief's ``awk`` rule, in Python so it can fail a test: any
    ``self.call_after_refresh(`` within ``window`` lines after a line that
    hands the rebuild to the VIEWER's message pump. Those two pumps have no
    ordering, so a follow-up queued that way runs against whichever tree it
    happens to find.
    """
    source = Path(library_screen_module.__file__).read_text(encoding="utf-8")
    anchor: int | None = None
    hits: list[tuple[int, str]] = []
    for number, line in enumerate(source.splitlines(), start=1):
        if _VIEWER_RECOMPOSE_ANCHOR.search(line):
            anchor = number
            continue
        if anchor is None or number - anchor > window:
            continue
        if "self.call_after_refresh(" in line:
            hits.append((number, line.strip()))
    return hits


def test_no_viewer_recompose_follow_up_rides_the_screen_pump():
    """task-31950 AC#2: every one of them goes through the viewer seam.

    On the merge-base this returns the two ``_sync_library_media_viewer_
    state`` tail sites (the mutation gate and the progress restore).
    """
    hits = _screen_pump_follow_ups_after_a_viewer_recompose()
    assert hits == [], (
        "These follow-ups still ride the screen's pump after a viewer "
        f"recompose; route them through the viewer seam: {hits}"
    )


@pytest.mark.asyncio
async def test_edit_save_mounts_already_gated_while_a_media_write_is_in_flight():
    """task-31950: the mutation gate has to see the RECOMPOSED Save button.

    Opening the edit form while a Media write holds the interlock is the
    one gesture that mounts a Save the gate has not already been applied
    to. The gate was queued with ``screen.call_after_refresh`` from the
    sync's tail, so it ran while the viewer's new children were not mounted
    yet -- ``query_one`` raised ``NoMatches``, the gate returned, and the
    form came up with a live Save on top of an unsettled write.
    """
    host = _four_action_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        await _open_reader_more(screen, pilot)

        # The shared write interlock, taken exactly as
        # ``_run_library_media_mutation`` takes it.
        screen._media_state.bulk_delete_in_flight = True
        screen.query_one("#library-media-edit", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-edit-save")
        # Let both pumps drain: the gate is a one-shot, so a settle window
        # is the honest wait (a poll would pass the moment it happened to
        # land and hide the ordering this pins).
        for _ in range(4):
            await pilot.pause()

        save = screen.query_one("#library-media-edit-save", Button)
        assert save.is_attached
        assert save.disabled, (
            "The edit form mounted a live Save while a Media write was in "
            f"flight (label={save.label!r})."
        )


@pytest.mark.asyncio
async def test_returning_to_read_restores_the_reading_position_once():
    """task-31954 AC#2: ONE owner schedules the restore for a mode change.

    ``handle_library_media_reader_mode`` schedules it through the viewer
    seam AND used to re-arm ``_sync_library_media_viewer_state``'s own
    arm-once guard by nulling ``_media_state.progress_restored_id``, so a
    single Analysis -> Read press restored twice. Idempotent today only
    because the restore is a ``scroll_to``.
    """
    host = _four_action_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        loaded_id = screen._media_state.reader_session.loaded_id
        assert loaded_id is not None
        screen._media_state.read_scroll_by_id[loaded_id] = (0, 3)

        restores: list[str] = []
        real_restore = screen._restore_library_media_loaded_progress

        def counting_restore(expected_id: str) -> None:
            restores.append(expected_id)
            real_restore(expected_id)

        screen._restore_library_media_loaded_progress = counting_restore

        screen.query_one("#library-media-reader-select-analysis", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: screen._media_state.reader_session.mode == "analysis",
            message="The Reader never switched to Analysis.",
        )
        for _ in range(4):
            await pilot.pause()
        restores.clear()

        screen.query_one("#library-media-reader-select-read", Button).press()
        await _wait_for_condition(
            pilot,
            lambda: restores,
            message="Returning to Read never restored the reading position.",
        )
        for _ in range(4):
            await pilot.pause()

        assert restores == [loaded_id], (
            f"one Analysis -> Read press scheduled {len(restores)} restores: "
            f"{restores}"
        )


@pytest.mark.parametrize(
    ("items_width", "expected"),
    [(36, "document · updated 5m · a…"), (44, _ANALYSED_SECONDARY)],
)
@pytest.mark.asyncio
async def test_analysed_secondary_is_bounded_at_custom_items_widths(
    items_width, expected
):
    """Pin truncation at the floor and full text at a wider custom size.

    A custom 36-cell Items pane gives its canvas 32 cells after padding.
    Since task-32060 the canvas fits that content box instead of overflowing
    it with its own 36-cell minimum. The longer labelled age from task-32347
    now ellipsises the analysis suffix at that floor; a wider pane shows it.
    """
    host = _review_state_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._media_state.reader_preferences = dataclasses.replace(
            screen._media_state.reader_preferences,
            custom_widths_enabled=True,
            items_width=items_width,
        )
        screen._sync_library_media_reader_layout_from_shell()
        await _wait_for_condition(
            pilot,
            lambda: _items_pane_width(screen) == items_width - 4,
            message=lambda: (
                f"Items canvas width={_items_pane_width(screen)}; "
                f"view={screen._media_state.view}; "
                f"preferences={screen._media_state.reader_preferences}; "
                f"layout={screen._media_state.reader_layout}"
            ),
        )

        # The crop at this width clips a neighbouring pane border into the
        # right edge, so strip that too before comparing the row's own text.
        _titles, secondaries = _painted_media_rows(host, screen)
        assert [line.strip(" │") for line in secondaries[:2]] == [
            expected,
            expected,
        ], secondaries
        assert all("…" not in line for line in secondaries[2:]), secondaries


# ---------------------------------------------------------------------------
# task-28008 (critique #5 P2): a keyword-only hit says which keyword matched.
# ---------------------------------------------------------------------------


def _match_reason_items() -> list[dict]:
    """Three ``article`` rows, all aged "2m", for the query ``notes``.

    Row 1 matches ONLY through its keyword; row 2 matches through its title
    (so it explains itself and earns no reason); row 3 matches only through
    a keyword too long for the Items pane's 36-cell floor.
    """
    now = datetime.now(timezone.utc)
    return [
        {
            "id": "media-1",
            "title": "Opening remarks",
            "type": "article",
            "last_modified": (now - timedelta(minutes=2, seconds=10)).isoformat(),
            "keywords": ["notes"],
            "content": "Transcript of the opening session.",
            "version": 1,
        },
        {
            "id": "media-2",
            "title": "Field notes",
            "type": "article",
            "last_modified": (now - timedelta(minutes=2, seconds=20)).isoformat(),
            "keywords": [],
            "content": "A body about nothing in particular.",
            "version": 1,
        },
        {
            "id": "media-3",
            "title": "Closing remarks",
            "type": "article",
            "last_modified": (now - timedelta(minutes=2, seconds=30)).isoformat(),
            "keywords": ["notesandmorestuff"],
            "content": "Transcript of the closing session.",
            "version": 1,
        },
    ]


@pytest.mark.parametrize(
    ("size", "expected"),
    [
        (
            (235, 52),
            [
                "article · updated 2m · keyword: notes · loaded",
                "article · updated 2m",
                "article · updated 2m · keyword: notesandmo…",
            ],
        ),
        (
            (100, 30),
            [
                "article · updated 2m · keyword: notes…",
                "article · updated 2m",
                "article · updated 2m · keyword: notes…",
            ],
        ),
    ],
    ids=["wide", "narrow"],
)
@pytest.mark.asyncio
async def test_keyword_only_rows_paint_the_keyword_that_matched(size, expected):
    """The reason is WORDS on the secondary line, and only where it is needed.

    ``Field notes`` matched the query in its own painted title, so it says
    nothing extra; the two rows whose match lives in a keyword name it. The
    keyword is capped at ten characters because the Items pane's floor is
    36 cells -- ``notesandmorestuff`` would push the line past it whole.

    task-32347 made both sizes tighter and the expectations are now
    per-size rather than shared: labelling the age ("updated 2m") costs 8
    cells on every row, so the narrow pane no longer has room for the
    trailing "· loaded" on the row the Reader holds. The term itself still
    paints -- dropping the review round's " ago" is what bought that back.
    """
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_match_reason_items())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "notes")
        for _ in range(3):
            await pilot.pause()

        lines = _painted_item_lines(host, screen)
        secondaries = [line.strip() for line in lines if "article · " in line]
        # task-32347 cost, pinned rather than hidden: labelling the age
        # ("updated 2m", 10 cells where "2m" was 2) made every secondary 8
        # cells longer, and task-32364 AC#1 adds "· loaded" to whichever
        # row the Reader holds. At the wide size both still fit; at the
        # narrow one the "· loaded" suffix is what runs out of room.
        assert secondaries == expected, secondaries



@pytest.mark.asyncio
async def test_keyword_reason_clips_at_the_36_cell_items_floor():
    """Fix round 1 (1): what the floor actually does, pinned honestly.

    The ten-character cap keeps the line SHORT; it does not make it fit
    here. A 36-cell Items pane spends 4 cells on its own padding, so the
    row has ~28 cells and `article · updated 2m · keyword: notes` needs 37 --
    at the floor a keyword row still cannot show its whole term, for the
    short keyword as well as the long one. The neighbouring
    `test_analysed_secondary_is_bounded_at_custom_items_widths` also pins
    truncation at this floor and complete text at a wider custom size.

    task-32060: what changed is HOW it stops. The canvas used to carry a
    36-cell min-width, so at a 36-cell pane it overflowed its 32-cell slot
    and the row was cut mid-word by the pane edge, with no ellipsis and no
    way to tell truncation from the real value. The canvas now takes the
    slot it is given, so the row ellipsises inside it.
    """
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=_match_reason_items())
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "notes")
        screen._media_state.reader_preferences = dataclasses.replace(
            screen._media_state.reader_preferences,
            custom_widths_enabled=True,
            items_width=36,
        )
        screen._sync_library_media_reader_layout_from_shell()
        await _wait_for_condition(
            pilot,
            # The canvas fills the pane's 32-cell content box (36 less its
            # own 4 cells of padding) instead of overflowing it (task-32060).
            lambda: _items_pane_width(screen) == 32,
            message="The Items pane never reached its 36-cell floor.",
        )
        for _ in range(3):
            await pilot.pause()

        lines = _painted_item_lines(host, screen)
        # The crop at this width clips a neighbouring pane border into the
        # right edge, so strip that before comparing the row's own text.
        secondaries = [
            line.strip(" │") for line in lines if "article · " in line
        ]
        # task-32347 costs the floor 6 cells, and the pin says exactly how
        # far that reaches rather than softening: the row still gets INTO
        # its "keyword:" label and no further. (The review round's rejected
        # "updated 2m ago" spent 4 more and cut the label off entirely --
        # that is what the shorter grammar bought back.) Recorded here so
        # nobody re-derives "the cap makes it fit" from a pin that never
        # said so.
        assert secondaries == [
            "article · updated 2m · ke…",
            "article · updated 2m",
            "article · updated 2m · ke…",
        ], secondaries
        # The row without a reason still paints whole -- the shortfall is the
        # suffix's own cost, not a regression in the base secondary line.
        assert "article · updated 2m" in secondaries, secondaries


@pytest.mark.asyncio
async def test_flag_keyword_reason_paints_no_half_flag():
    """task-32044 (crit #7 P2): a flag keyword never paints a lone indicator.

    A regional-indicator flag is a PAIR painted as one 2-cell glyph. The
    ten-cell cut could split the pair at the tail, leaving a lone indicator a
    real terminal paints as a 2-cell box -- the finding's +2 row-frame drift
    (border at 237 vs 235). (Textual's headless compositor measures cells the
    way rich does, so it cannot reproduce the terminal-font width itself; what
    it pins is that the production render path never emits the half-flag that
    causes the drift.) So the painted reason carries only whole flags: an even
    number of regional-indicator code points.
    """
    flag_jp = "\U0001F1EF\U0001F1F5"
    flag_us = "\U0001F1FA\U0001F1F8"
    # "flagx" is five cells, so the ten-cell cut lands mid-pair without the
    # fix, leaving a lone indicator; the title carries no "flagx" so the row
    # can only match through the keyword.
    keyword = "flagx" + (flag_jp + flag_us) * 3
    items = _match_reason_items()
    items[0]["keywords"] = [keyword]  # row 1 now matches only via the flag keyword
    app = _build_media_test_app()
    _seed_conversations(app, _two_conversations(), media=items)
    host = LibraryProductionCSSHarness(app)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "flagx")
        for _ in range(3):
            await pilot.pause()

        lines = _painted_item_lines(host, screen)
        reason = "".join(line for line in lines if "keyword: flagx" in line)
        assert reason, lines
        ri = sum(1 for ch in reason if 0x1F1E6 <= ord(ch) <= 0x1F1FF)
        assert ri >= 2, reason  # at least one whole flag actually paints
        assert ri % 2 == 0, (ri, reason)  # no lone half-flag


@pytest.mark.asyncio
async def test_viewer_sync_follow_up_chains_the_restore_when_its_target_is_gone():
    """PR F's restore is CHAINED behind the follow-up, never evicted.

    ``queue_after_recompose`` REPLACES, and the sync queues task-31567's
    focus restore on that same one slot. The helper captures it and calls it
    after its own target, so a follow-up whose target is not composed leaves
    focus where the restore puts it (the captured identity) -- never on the
    pane grip Textual re-picks when the focused child is recomposed away.
    """
    host = _four_action_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        screen.query_one("#library-media-reader-find", Button).focus()
        await pilot.pause()

        screen._media_state.reader_session = set_more_open(
            screen._media_state.reader_session, True
        )
        screen._after_library_media_viewer_sync("#library-media-reader-absent")
        await _wait_for_selector(screen, pilot, "#library-media-reader-more-actions")

        await _settle_focus_on(
            screen,
            pilot,
            "library-media-reader-find",
            "An absent follow-up target",
        )
        assert not screen.focused.has_class(LIBRARY_ADAPTIVE_READER_GRIP_CLASS)


# ---------------------------------------------------------------------------
# task-31635 (critique #5 items 12, 14): the empty Reader under a failed list,
# and the rendered Markdown H1's alignment against the reading column.
# ---------------------------------------------------------------------------


def _first_glyph_column(host, widget) -> int:
    """Absolute column of the first painted glyph inside ``widget``."""
    region = widget.region
    strips = list(host.screen._compositor.render_strips())
    for y in range(region.y, min(region.bottom, len(strips))):
        row = strips[y].crop(region.x, region.right).text
        if row.strip():
            return region.x + len(row) - len(row.lstrip())
    raise AssertionError(f"Nothing painted inside {widget!r} at {region}.")


async def _open_media_with_a_failed_first_page(host, pilot, exc: BaseException):
    """Open Media with its FIRST page failing, so no rows are ever retained.

    ``_force_media_page_failure`` fails a page that already applied, which
    keeps ``retained_items`` -- the state where rows stay painted and
    pressable. This is the other one: nothing was ever applied, so there is
    genuinely nothing to select.
    """

    async def _fails(**_kwargs):
        raise exc

    host.app_instance.media_reading_scope_service.search_media = _fails
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-media").press()
    controller = screen._library_media_browse_controller
    await _wait_for_condition(
        pilot,
        lambda: (
            controller.failure is not None
            and not controller.loading
            and not controller.retained_items
            and bool(screen.query("#library-media-load-failure-copy"))
        ),
        message="The first-page Media failure never settled.",
    )
    await pilot.pause()
    return screen


@pytest.mark.asyncio
async def test_empty_reader_placeholder_names_a_failed_list():
    """task-31635 (critique #5 item 12): "Select a media item" was a lie.

    With the FIRST list load failed there is nothing to select, and the
    empty Reader still invited the user to select something -- the only
    line on screen saying so while the callout beside it said the load
    had failed.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_with_a_failed_first_page(
            host, pilot, sqlite3.OperationalError("database is locked")
        )

        assert not screen.query(".library-media-row")
        empty = screen.query_one("#library-media-reader-empty", Static)
        assert str(empty.content) == "Nothing loaded — the list could not be loaded."
        assert "Nothing loaded" in _painted(host, empty.region)


@pytest.mark.asyncio
async def test_page_failure_that_retains_rows_keeps_the_select_invitation():
    """task-31635 fix round 1: retained rows ARE selectable, so say so.

    A page-1 failure after page 1 applied keeps every row painted, enabled
    and pressable (the recovery callout's whole point). Telling the reader
    "the list could not be loaded" there contradicts the two rows it can
    still open.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        empty = screen.query_one("#library-media-reader-empty", Static)
        assert str(empty.content) == "Select a media item to read it here."

        await _force_media_page_failure(
            host, screen, pilot, sqlite3.OperationalError("database is locked")
        )

        rows = list(screen.query(".library-media-row"))
        assert len(rows) == 2
        assert not any(row.disabled for row in rows)
        empty = screen.query_one("#library-media-reader-empty", Static)
        assert str(empty.content) == "Select a media item to read it here."
        assert "Select a media item to read it here." in _painted(host, empty.region)


def _h1_markdown_host() -> LibraryProductionCSSHarness:
    """Two Markdown items whose first line is an H1 heading."""
    app = _build_media_test_app()
    items = [
        {
            "id": f"media-{index}",
            "title": f"Quarterly notes {index}",
            "type": "markdown",
            "last_modified": "2026-07-06T10:00:00Z",
            "content": (
                "# Quarterly budget\n\n"
                "The reading column starts at the left edge of the box.\n"
            ),
            "version": 1,
        }
        for index in (1, 2)
    ]
    _seed_conversations(app, _two_conversations(), media=items)
    return LibraryProductionCSSHarness(app)


@pytest.mark.asyncio
async def test_rendered_markdown_h1_starts_in_the_body_column():
    """task-31635 (critique #5 item 14): the H1 aligns with the prose.

    Textual's ``MarkdownH1`` default is ``content-align: center middle``,
    so a document title floated to the middle of the 92-cell reading
    measure while every other line began at its left edge -- the heading
    read as a banner detached from the text it introduces.
    """
    from textual.widgets._markdown import MarkdownH1, MarkdownParagraph

    host = _h1_markdown_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)

        markdown = screen.query_one("#library-media-viewer-content-markdown")
        heading = markdown.query_one(MarkdownH1)
        body = markdown.query_one(MarkdownParagraph)

        assert "Quarterly budget" in _painted(host, heading.region)
        assert _first_glyph_column(host, heading) == _first_glyph_column(host, body), (
            heading.region,
            body.region,
            _painted(host, heading.region),
        )


# ---------------------------------------------------------------------------
# task-31635 (critique #5 item 6): with the FIRST list load failed and
# nothing behind it, "Export…" -- which exports the whole filtered list --
# stayed live and colour-normal beside the failure callout, while "Select"
# next to it already rendered its "○" marker and said why.
#
# Fix round 1 narrows it to the same predicate the empty Reader uses
# (`_library_media_list_unselectable`): a failure with NO retained rows. A
# later-page failure keeps its rows -- that retention is the callout's whole
# point -- and those rows export fine. "Trash" is never gated: it is a route
# into a view with its own fetch, callout and Retry, and disabling it would
# remove the only way to reach deleted items exactly when the store is
# unhappy.
#
# task-31960: "Review these" -- the other whole-filtered-list action -- was
# the one left outside that gate, so it stood live and colour-normal beside
# a dimmed Export on the same failed first page. Symmetry was one line, so
# both are pinned here together.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
async def test_failed_first_page_gates_export_with_its_reason(size):
    """A first load that failed with nothing behind it disables the two
    whole-filtered-list actions -- Export… and "Review these" (task-31960).
    """
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_with_a_failed_first_page(
            host, pilot, sqlite3.OperationalError("database is locked")
        )

        export = screen.query_one("#library-media-export", Button)
        assert export.disabled
        assert str(export.label) == "○ Export…"
        assert str(export.tooltip) == "Couldn't load media · database is locked."
        assert "○" in _painted(host, export.region), _painted(host, export.region)

        # task-31960: the same gate, the same reason, the same marker --
        # "Review these" pins the whole filtered list as a review set, and
        # there is no list to pin.
        review = screen.query_one("#library-media-review", Button)
        assert review.disabled
        assert str(review.label) == "○ Review these"
        assert str(review.tooltip) == "Couldn't load media · database is locked."
        assert "○" in _painted(host, review.region), _painted(host, review.region)

        # Trash stays the live route into the deleted items.
        trash = screen.query_one("#library-media-trash-open", Button)
        assert not trash.disabled
        assert str(trash.label) == "Trash"
        assert "Trash" in _painted(host, trash.region)


@pytest.mark.asyncio
async def test_page_failure_that_retains_rows_leaves_export_live():
    """Fix round 1's negative control: retained rows export fine."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _force_media_page_failure(
            host, screen, pilot, sqlite3.OperationalError("database is locked")
        )

        # The callout IS up (the broad predicate this used to gate on)...
        assert screen.query("#library-media-load-failure-copy")
        assert screen._library_media_browse_controller.failure is not None
        assert len(screen.query(".library-media-row")) == 2
        # ...and Export… is still live over the rows that survived it.
        export = screen.query_one("#library-media-export", Button)
        assert not export.disabled
        assert str(export.label) == "Export…"
        # task-31960: so is "Review these" -- retained rows are reviewable.
        review = screen.query_one("#library-media-review", Button)
        assert not review.disabled
        assert str(review.label) == "Review these"


@pytest.mark.asyncio
async def test_a_healthy_list_leaves_export_and_trash_live():
    """The other negative control: no failure, no gate (task-31635 item 6)."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        export = screen.query_one("#library-media-export", Button)
        trash = screen.query_one("#library-media-trash-open", Button)
        review = screen.query_one("#library-media-review", Button)
        assert not export.disabled
        assert not trash.disabled
        assert not review.disabled
        assert str(export.label) == "Export…"
        assert str(trash.label) == "Trash"
        assert str(review.label) == "Review these"


# --- task-31635 Task 3 (critique #5 items 7/8/11/19 + Qodo 18) ----------------


@pytest.mark.asyncio
async def test_raising_follow_up_still_runs_the_queued_focus_restore():
    """Qodo on #2473 (item 18): the chained restore rides a ``finally``.

    ``_after_library_media_viewer_sync`` chains its own follow-up ahead of
    whatever the sync already queued -- PR F's task-31567 focus restore. A
    follow-up that raises (the scroll-progress restore is one) dropped that
    restore on the floor and left focus wherever the recompose put it. The
    exception must still surface; only the ordering guarantee changes.
    """
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        viewer = screen.query_one("#library-media-viewer")
        # Stand on a child the viewer owns so the sync's own restore has
        # something to put back.
        screen.query_one("#library-media-reader-more", Button).focus()
        await pilot.pause()
        # A real compose-input flip, so the sync genuinely recomposes and
        # queues task-31567's restore for us to chain behind.
        screen._media_state.reader_session = set_more_open(
            screen._media_state.reader_session, True
        )

        def boom() -> None:
            raise RuntimeError("follow-up blew up")

        with pytest.raises(RuntimeError, match="follow-up blew up"):
            screen._after_library_media_viewer_sync(boom)
            chained = viewer._post_recompose_callback
            assert chained is not None
            chained()

        await pilot.pause()
        await pilot.pause()
        focused = screen.focused
        assert focused is not None
        assert focused is viewer or viewer in focused.ancestors, focused


@pytest.mark.asyncio
async def test_saving_an_analysis_marks_its_row_analysed_without_a_refetch():
    """Qodo on #2475 (item 19): the row follows the analysis it just gained.

    ``analysed`` is projected in SQL and frozen into the retained row at
    browse-state build, so a freshly saved analysis left its own row
    unmarked until the next page fetch -- the one row the user just proved
    has an analysis. The save seam re-reads that ONE row from the same SQL
    projection (an id-scoped re-fetch on a human-paced gesture, never on the
    page path) and the existing canvas patch paints it.

    It asks the projection rather than trusting the write's own claim,
    because live on 2026-09-07 those disagreed -- see
    ``_reproject_library_media_analysis_row``.
    """
    host = _review_state_host(count=4, analysed=0)
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        for _ in range(3):
            await pilot.pause()
        _titles, before = _painted_media_rows(host, screen)
        assert [line.strip() for line in before] == ["document · updated 5m"] * 4, before

        screen.query_one("#library-media-row-0", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-reader-select-analysis")
        screen.query_one("#library-media-reader-select-analysis", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit")
        screen.query_one("#library-media-analysis-edit", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-analysis-edit-text")
        screen.query_one(
            "#library-media-analysis-edit-text", TextArea
        ).text = "A brand new analysis"
        searches_before = len(host.app_instance.media_reading_scope_service.search_calls)
        screen.query_one("#library-media-analysis-save", Button).press()

        await _wait_for_condition(
            pilot,
            lambda: bool(host.app_instance.media_reading_scope_service.analysis_calls)
            and not screen._media_state.editing_analysis,
            message="The analysis save never completed.",
        )
        for _ in range(3):
            await pilot.pause()

        _titles, after = _painted_media_rows(host, screen)
        assert [line.strip() for line in after] == [
            # task-32364 AC#1: this row is the one open in the Reader, so
            # it carries the state word on its fact line.
            f"{_ANALYSED_SECONDARY} · loaded",
            "document · updated 5m",
            "document · updated 5m",
            "document · updated 5m",
        ], after
        # ...and the only read it cost was one id-scoped row, never a
        # re-page of the list.
        extra = host.app_instance.media_reading_scope_service.search_calls[
            searches_before:
        ]
        assert len(extra) == 1, extra
        assert extra[0]["id_allowlist"] == [1], extra[0]
        assert extra[0]["limit"] == 1, extra[0]


async def _apply_media_filter(screen, pilot, query: str) -> None:
    """Type ``query`` into the Media filter and settle the authoritative page."""
    screen.query_one("#library-media-filter", Input).value = query
    await _wait_for_condition(
        pilot,
        lambda: (
            screen._library_media_browse_controller.applied_scope is not None
            and screen._library_media_browse_controller.applied_scope.query == query
            and not screen._library_media_browse_controller.loading
        ),
        message=f"The filter {query!r} never applied.",
    )
    for _ in range(3):
        await pilot.pause()


@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
@pytest.mark.asyncio
async def test_zero_result_filter_keeps_the_sets_entry(size):
    """Item 7: Sets is navigation, not a result.

    The fresh-empty page distils to exactly ONE recovery action, and the
    Sets opener was composed under that same gate -- so filtering to zero
    rows took away the only route back to a saved review set at the moment
    the list had nothing else to offer. The picker opens over any list
    (it carries its own empty copy and "Read later"), so it is never
    disabled here.
    """
    host = _review_state_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "zzz-nothing-matches")

        assert not screen.query(".library-media-row")
        sets = screen.query_one("#library-media-review-sets", Button)
        assert sets.display is True
        assert not sets.disabled
        assert "Sets" in _painted(host, sets.region)


@pytest.mark.asyncio
async def test_a_single_result_filter_announces_that_enter_opens_it():
    """Item 8 (declined, announced): the filter's first hit auto-loads.

    Selecting the first authoritative result is intentional and pinned
    (``test_filter_uses_authoritative_search_and_restores_page_three_anchor``
    in Tests/UI/test_library_media_reader_flow.py -- and clearing the
    filter restores the previous anchor, the symmetric half). So the list
    says what happened instead of changing it.
    """
    host = _review_state_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _apply_media_filter(screen, pilot, "Doc 3")

        assert len(screen.query(".library-media-row")) == 1
        status = screen.query_one("#library-media-status", Static)
        assert status.display is True
        assert str(status.renderable) == "1 result · Enter opens"
        assert "1 result · Enter opens" in _painted(host, status.region)

        # Two hits say nothing -- this line only explains the single-hit
        # auto-load.
        await _apply_media_filter(screen, pilot, "Doc")
        assert len(screen.query(".library-media-row")) == 4
        assert screen.query_one("#library-media-status", Static).display is False


@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
@pytest.mark.asyncio
async def test_the_reader_says_its_item_is_not_in_the_open_trash_list(size):
    """Item 11: the Reader beside Trash still holds a LIVE media item.

    Opening Trash swaps the Items pane for the deleted-items list and leaves
    the Reader on whatever was open -- a live item sitting beside a list of
    deleted ones, with nothing saying which list it came from. The cheaper
    honest option of the two on offer: one identity line, in the slot the
    server-item line already uses (task-31277's grammar), instead of
    clearing the Reader -- clearing would throw away the reading position
    the user comes back to.
    """
    host = _review_state_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        assert not screen.query("#library-media-reader-identity")

        screen.query_one("#library-media-trash-open", Button).press()
        await _wait_for_selector(screen, pilot, "#library-media-trash-canvas")
        for _ in range(3):
            await pilot.pause()

        identity = screen.query_one("#library-media-reader-identity", Static)
        assert str(identity.renderable) == "Showing a Media item · not in Trash"
        assert "not in Trash" in _painted(host, identity.region)


@pytest.mark.parametrize("size", [(235, 52), (100, 30)], ids=["wide", "narrow"])
@pytest.mark.asyncio
async def test_more_row_actions_share_one_grid_column_grammar(size):
    """Item 16 (declined, shipped by PR H #2470): the indent is the grid's.

    "Open manager" used to sit one cell further in than its siblings. PR H
    replaced the More disclosure's bare Vertical with a single ``ItemGrid``
    (``#library-media-reader-more-actions``, now fixed 17-cell columns), so
    every action in the row is laid out on the same column origins by
    construction. This is the painted proof, kept as a pin so the column
    grammar cannot silently drift back.
    """
    host = _review_state_host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        screen.query_one("#library-media-reader-more", Button).press()
        grid = await _wait_for_selector(
            screen, pilot, "#library-media-reader-more-actions"
        )
        for _ in range(2):
            await pilot.pause()

        actions = list(grid.query(Button))
        assert [action.id for action in actions] == [
            "library-media-edit",
            "library-media-open",
            "library-media-delete",
        ], actions
        # task-31980 deliberately separates the destructive label by two
        # cells inside its grid slot. Compare slot origins, then verify the
        # full labels are still painted rather than treating that margin as
        # a broken column alignment.
        expected_margins = {
            "library-media-edit": 0,
            "library-media-open": 0,
            "library-media-delete": 2,
        }
        assert {
            action.id: action.styles.margin.left for action in actions
        } == expected_margins
        origins = {
            action.id: action.region.x - expected_margins[action.id]
            for action in actions
        }
        columns = sorted(set(origins.values()))
        # One pitch for the whole row: "Open manager" starts exactly one
        # column after "Edit metadata" and one before "Move to trash",
        # never on an origin (or an extra cell of indent) of its own.
        pitches = {second - first for first, second in zip(columns, columns[1:])}
        assert len(pitches) == 1, [(action.id, action.region) for action in actions]
        # ...and every row of the grid starts at the same leftmost column.
        rows: dict[int, list[int]] = {}
        for action in actions:
            rows.setdefault(action.region.y, []).append(origins[action.id])
        assert {min(xs) for xs in rows.values()} == {columns[0]}, rows
        painted = _painted(host, grid.region)
        for label in ("Edit metadata", "Open manager", "Move to trash"):
            assert label in painted, painted


# ---------------------------------------------------------------------------
# task-31956: the reviewed decoration costs O(visible rows), not O(active set)
# ---------------------------------------------------------------------------


def _count_set_loads(service) -> list[str]:
    """Count whole-set loads through ``service``, returning the tally list."""
    loads: list[str] = []
    inner = service.get_active_review_set

    def counted():
        review_set = inner()
        loads.append("" if review_set is None else review_set.set_id)
        return review_set

    service.get_active_review_set = counted
    return loads


async def _decoration_fixture(host, pilot):
    """A settled Media list with a two-item active set over four rows.

    The load tally starts counting BEFORE the canvas build that follows the
    create, so it covers the first real decoration rather than a cache the
    fixture already warmed.
    """
    screen = await _open_media_list(host, pilot)
    service = screen._review_set_service()
    set_id = service.create_review_set(
        "These", origin="browse", items=[(1, "Doc 1"), (2, "Doc 2")]
    )
    loads = _count_set_loads(service)
    _sync_library_canvas(screen, "media")
    for _ in range(3):
        await pilot.pause()
    items = screen._library_media_browse_controller.retained_items
    assert len(items) == 4, items
    return screen, service, set_id, items, loads


def _reviewed(rows) -> list[bool | None]:
    return [row["reviewed"] for row in rows]


@pytest.mark.asyncio
async def test_decorating_a_page_loads_the_active_set_once_not_per_build():
    """task-31956 AC#1: an unchanged set is read once, not per canvas build.

    ``get_active_review_set`` loads the header AND every pinned item row (up
    to ``REVIEW_SET_CAP`` = 500) and the decoration ran it at every one of
    the ~30 viewer-flip sync sites, to stamp at most a page of rows. The
    map is now cached against the service's write revision, so repeated
    builds cost the rows they decorate and nothing else.
    """
    host = _review_state_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen, _service, _set_id, items, loads = await _decoration_fixture(
            host, pilot
        )
        # The canvas build inside the fixture already decorated this page.
        assert len(loads) == 1, loads

        first = screen._decorate_library_media_reviewed(items)
        for _ in range(20):
            screen._decorate_library_media_reviewed(items)

        assert len(loads) == 1, loads
        assert _reviewed(first) == [False, False, None, None], first


@pytest.mark.asyncio
async def test_marking_an_item_done_invalidates_the_decoration_cache():
    """task-31956 AC#2: the mark seam is a write, so the next page is right.

    Every write goes through the service's one transaction helper, which is
    what the cache keys off -- so this holds for the ``m`` gesture, the
    walker's auto-mark, and a direct service call alike.
    """
    host = _review_state_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen, service, set_id, items, loads = await _decoration_fixture(
            host, pilot
        )
        assert len(loads) == 1, loads

        assert _reviewed(screen._decorate_library_media_reviewed(items)) == [
            False,
            False,
            None,
            None,
        ]
        service.mark_item_done(set_id, backing_media_id=1, done=True)
        marked = screen._decorate_library_media_reviewed(items)

        assert _reviewed(marked) == [True, False, None, None], marked
        assert len(loads) == 2, loads


@pytest.mark.asyncio
async def test_leaving_and_re_entering_review_invalidates_the_decoration_cache():
    """task-31956 AC#2: activation is a write too -- the markers follow it."""
    host = _review_state_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen, service, set_id, items, _loads = await _decoration_fixture(
            host, pilot
        )

        service.deactivate_active()
        assert _reviewed(screen._decorate_library_media_reviewed(items)) == [
            None,
            None,
            None,
            None,
        ]

        service.activate(set_id)
        assert _reviewed(screen._decorate_library_media_reviewed(items)) == [
            False,
            False,
            None,
            None,
        ]


@pytest.mark.asyncio
async def test_a_write_landing_during_the_load_is_not_stamped_as_included():
    """Fix round 1: the stamp is the revision read BEFORE the load.

    `dismiss`/`undismiss` commit on a thread (`asyncio.to_thread`) while
    this thread decorates, so a write can land between the read of the set
    and the stamp. Stamping the revision read AFTER the load would claim
    that write was included, freezing a map that no later sync repairs --
    the rows would keep painting marks the set no longer has. Stamping the
    earlier value only costs one extra read.
    """
    host = _review_state_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen, service, set_id, items, _loads = await _decoration_fixture(
            host, pilot
        )

        healthy = service.get_active_review_set

        def racing():
            # The snapshot this call will cache...
            review_set = healthy()
            # ...and a write that commits (and bumps) before it is stamped.
            service.mark_item_done(set_id, backing_media_id=1, done=True)
            return review_set

        # Invalidate first, so the racing build is the one that loads.
        service.set_cursor(set_id, 0)
        service.get_active_review_set = racing
        stale = screen._decorate_library_media_reviewed(items)
        assert _reviewed(stale) == [False, False, None, None], stale

        service.get_active_review_set = healthy
        fresh = screen._decorate_library_media_reviewed(items)
        assert _reviewed(fresh) == [True, False, None, None], fresh


@pytest.mark.asyncio
async def test_a_storage_error_is_never_cached_as_no_active_set():
    """task-30042 doctrine survives the cache: it fails OPEN, and forgets.

    A cached failure would cost the markers for the rest of the session on
    one transient read error, so the failure is what is NOT remembered --
    every build retries while the cache is invalid, and the markers return
    with the storage.
    """
    host = _review_state_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen, service, set_id, items, _loads = await _decoration_fixture(
            host, pilot
        )
        # A write invalidates the cache, so the next decoration must read.
        service.mark_item_done(set_id, backing_media_id=1, done=True)

        attempts: list[str] = []
        healthy = service.get_active_review_set

        def boom():
            attempts.append("read")
            raise sqlite3.OperationalError("database is locked")

        service.get_active_review_set = boom
        assert _reviewed(screen._decorate_library_media_reviewed(items)) == [
            None,
            None,
            None,
            None,
        ]
        assert _reviewed(screen._decorate_library_media_reviewed(items)) == [
            None,
            None,
            None,
            None,
        ]
        assert len(attempts) == 2, attempts

        service.get_active_review_set = healthy
        assert _reviewed(screen._decorate_library_media_reviewed(items)) == [
            True,
            False,
            None,
            None,
        ]


# ---------------------------------------------------------------------------
# task-31957: the preview pane names the analysis state the row names
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row_index", "expected", "other"),
    [(0, "Analysed: yes", "Analysed: no"), (2, "Analysed: no", "Analysed: yes")],
)
@pytest.mark.asyncio
async def test_the_preview_pane_paints_the_selected_items_analysis_state(
    row_index: int, expected: str, other: str
):
    """task-31957: the pane answers for an analysed AND an un-analysed item.

    Painted on the real screen, over the real projection: rows 1-2 of
    ``_review_state_host`` carry an analysis and rows 3-4 do not, and the
    pane has to say which one the selection is on -- the row's line says
    "· analysed" or nothing, and until now the pane said neither.

    ``show_preview`` is flipped on for the assertion: since the permanent
    Reader shipped (``d99fb4a9c``) the screen passes ``show_preview=False``
    for every Media canvas path, so this pane is not on screen today -- see
    the batch-2 report. Everything else here is production: the screen's
    stylesheet, the pane's real geometry (a ~15-cell text measure inside
    the Items pane, which is why the line is kept short), and the state
    built by the same browse projection the rows come from.
    """
    host = _review_state_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        for _ in range(3):
            await pilot.pause()
        screen.query_one(f"#library-media-row-{row_index}", Button).press()
        for _ in range(3):
            await pilot.pause()

        canvas = screen.query_one("#library-media-canvas")
        canvas.show_preview = True
        await canvas.recompose()
        for _ in range(3):
            await pilot.pause()

        preview = screen.query_one("#library-media-preview")
        assert preview.region.area > 0, preview.region
        painted = _painted(host, preview.region)

        assert expected in painted, painted
        assert other not in painted, painted
        # The pane describes the item whose row is selected, not a
        # neighbour: its title is painted right above the answer.
        assert f"Doc {row_index + 1}" in painted, painted
