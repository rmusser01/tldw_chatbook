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

import pytest
from types import SimpleNamespace
from textual.widgets import Button, Input, OptionList, Static
from textual.worker import WorkerState

from tldw_chatbook.Library.library_media_reader_state import set_mode
from tldw_chatbook.UI.Screens import library_screen as library_screen_module
from tldw_chatbook.UI.Screens.library_screen import _sync_library_canvas
from tldw_chatbook.Widgets.AppFooterStatus import AppFooterStatus
from tldw_chatbook.Widgets.Library.library_media_reader_shell import (
    LibraryMediaReaderShell,
)

from Tests.UI.test_library_media_side_by_side import (
    _build_media_test_app,
    _open_media_list,
    _two_media_items,
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
    _seed_conversations,
    _submit_content_search_query,
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
        lambda: screen._library_media_reader_session.loaded_id == row_id,
        message=f"] never loaded row {expected_row}.",
    )
    await pilot.pause()
    return row_id


async def _switch_to_analysis(screen, pilot) -> None:
    screen._library_media_reader_session = set_mode(
        screen._library_media_reader_session, "analysis"
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
        assert screen._library_media_reader_session.mode == "analysis"
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
        assert screen._library_media_reader_session.mode == "analysis"

        await pilot.press("escape")
        await pilot.pause()
        await pilot.pause()
        assert not screen.query("#library-media-content-search-controls")
        assert screen._library_media_find_open is False


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
        assert screen._library_media_find_open is True
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
        assert screen._library_media_find_open is False


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
        screen._library_media_delete_receipt_ids = ("local:media:1",)
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
        screen._library_media_analyze_total = 40
        screen._library_media_analyze_done = 38
        screen._library_media_analyze_failed_ids = ("local:media:1", "local:media:2")
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
        screen._library_media_analyze_running = True
        screen._library_media_analyze_total = 40
        screen._library_media_analyze_done = 0
        screen._library_media_analyze_failed_ids = ("local:media:1", "local:media:2")
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
        screen._library_media_analyze_total = 40
        screen._library_media_analyze_done = 40
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
        screen._library_media_analyze_total = 3
        screen._library_media_analyze_done = 0
        screen._library_media_analyze_failed_ids = ("a", "b", "c")
        _sync_library_canvas(screen, "media")
        receipt = await _wait_for_selector(
            screen, pilot, "#library-media-analyze-receipt"
        )
        await pilot.pause()
        await pilot.pause()
        painted = _painted(host, receipt.region)
        assert "\u2715 analyzed \u00b7 0 of 3 \u00b7 3 failed" in painted, painted
        assert "\u2713" not in painted, painted
        assert "Retry failed" in painted, painted


@pytest.mark.asyncio
async def test_analyze_overwrite_choice_paints_both_options():
    """task-28007 AC#3: the already-analysed choice is armed IN the receipt row."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        screen._library_media_analyze_choice = (
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
        assert "1 of 2 already analysed" in painted, painted
        assert "—" not in painted, painted  # R3: no dangling dash
        assert "Skip them" in painted, painted
        assert "Overwrite" in painted, painted


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
                lambda: screen._library_media_analyze_running is False,
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
            screen._library_media_analyze_reason_cache = None
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

        assert screen._library_media_find_open is False
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

        assert screen._library_media_select_mode is True
        assert ("space", "toggle selection") in (
            screen._library_footer_shortcuts_for_current_state()
        )
        focused = screen.focused
        assert focused is not None and focused.has_class(
            "library-media-row"
        ), focused

        await pilot.press("space")
        await pilot.pause()
        assert screen._library_media_row_selection.count == 1


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
        assert screen._library_media_select_mode is True

        shell = screen.query_one(
            "#library-media-reader-shell", LibraryMediaReaderShell
        )
        before = shell.effective_layout
        shell.library_grip.focus()
        await pilot.pause()

        await pilot.press("space")
        await pilot.pause()
        await pilot.pause()

        shell = screen.query_one(
            "#library-media-reader-shell", LibraryMediaReaderShell
        )
        assert shell.effective_layout.library_open is before.library_open
        assert screen._library_media_row_selection.count == 0


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
        assert screen._library_media_select_mode is True

        def claim() -> bool | None:
            return screen.check_action("library_media_toggle_row_selection", ())

        # Entry lands on a row: Space is the selection key there.
        assert screen.focused.has_class("library-media-row"), screen.focused
        assert claim() is True

        shell = screen.query_one(
            "#library-media-reader-shell", LibraryMediaReaderShell
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
        assert screen._library_media_select_mode is False


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
        "#library-media-reader-shell", LibraryMediaReaderShell
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
        assert screen._library_media_view == "viewer"
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
    Escape) set ``_library_media_view = "list"`` while the three-pane
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
        assert screen._library_media_view == "viewer"
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
        assert screen._library_media_reader_session.more_open is False
        assert not screen.query("#library-media-reader-more-actions")


@pytest.mark.asyncio
async def test_f6_content_stop_is_visible_and_content_still_paints():
    """The F6 Reader stop tints the content box's own border (task-31221).

    Critique #4 (B cap_57): F6's first Reader candidate is the content
    scroller, but nothing on screen said so. The cue must be the EXISTING
    border, never an outline: the app-global ``*:focus`` outline paints
    OVER the outermost rows, which is why this asserts painted text too.
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
        assert focused_border[0] == unfocused[0] == "solid", (
            focused_border,
            unfocused,
        )
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
        assert screen._library_media_select_mode is True
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
                screen._library_media_select_mode is False
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
        assert screen._library_media_view == "viewer"


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
        assert screen._library_media_view == "viewer"

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
        assert screen._library_media_find_open is False
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
            """The five grip columns left of the Reader, top three rows."""
            viewer = screen.query_one("#library-media-viewer")
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
                _painted_row(host, y)[title.region.x - 5 : title.region.x]
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
    show the empty byline row task-31277 collapses. The type is ``pdf`` --
    outside ``_MARKDOWN_MEDIA_TYPES`` -- so no Rendered|Raw strip can enter
    the chrome count whatever the content sniffs as, and one deliberately
    long line proves the reading measure.
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
            # pdf is outside _MARKDOWN_MEDIA_TYPES, so no Rendered|Raw strip
            # can appear and the chrome count is independent of the sniff.
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
        assert box.region.width > 120, box.region
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
        assert str(generate.tooltip) == "No analysis provider is configured."
        # Belt and braces: the handler refuses with the same sentence.
        warnings: list[str] = []
        screen._notify_library_media_analysis_warning = warnings.append
        screen.handle_library_media_analysis_generate(
            SimpleNamespace(stop=lambda: None)
        )
        await pilot.pause()
        assert warnings == ["No analysis provider is configured."]
        assert screen._library_media_generating_analysis is False


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
