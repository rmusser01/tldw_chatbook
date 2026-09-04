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
from textual.widgets import Button, Input, OptionList

from tldw_chatbook.Library.library_media_reader_state import set_mode
from tldw_chatbook.UI.Screens.library_screen import _sync_library_canvas

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
    await _wait_for_selector(screen, pilot, "#library-media-viewer-analysis-title")
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


# ---------------------------------------------------------------------------
# task-31276 (critique #4 P2): the Find bar stays put; no join artifact.
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

    After Escape closed Find, a five-cell rule appeared immediately left of
    the Reader header's first text row ("Local Media item" then; the title
    since task-31277 made that identity line server-only) at the pane join,
    and persisted across later interactions (14 live captures; absent on a
    fresh open). The fresh-open row is the reference: nothing the Find
    gesture does may change it.
    """
    host = _find_host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_media_list(host, pilot)
        await _open_first_reader_row(screen, pilot)
        header = screen.query_one("#library-media-viewer-title")
        fresh_row = _painted_row(host, header.region.y)
        assert "Product Demo Video" in fresh_row, fresh_row

        async def _identity_row() -> str:
            await pilot.pause()
            await pilot.pause()
            widget = screen.query_one("#library-media-viewer-title")
            return _painted_row(host, widget.region.y)

        # Find opened then closed. The stray rule lands in the five pane-grip
        # columns immediately LEFT of the header Static, so the header's own
        # region cannot see it -- the whole row is the assertion.
        await _open_media_find(screen, pilot)
        await pilot.press("escape")
        after_find = await _identity_row()
        widget = screen.query_one("#library-media-viewer-title")
        assert "─" not in _painted(host, widget.region)
        join = after_find[widget.region.x - 5 : widget.region.x]
        assert "─" not in join, join
        assert after_find == fresh_row, (fresh_row, after_find)

        # A mode-tab click.
        screen.query_one("#library-media-reader-select-analysis", Button).press()
        after_tab = await _identity_row()
        assert after_tab == fresh_row, (fresh_row, after_tab)

        # The More menu opened then dismissed.
        screen.query_one("#library-media-reader-more", Button).press()
        await pilot.pause()
        await pilot.press("escape")
        after_more = await _identity_row()
        assert after_more == fresh_row, (fresh_row, after_more)


def _plain_local_host() -> LibraryProductionCSSHarness:
    """Two local items with neither an author nor a URL.

    ``_two_media_items`` carries an author on both rows, so it can never
    show the empty byline row task-31277 collapses. Content is plain prose
    (no markdown syntax) so the Rendered|Raw strip stays out of the chrome
    count, and one deliberately long line proves the reading measure.
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
            "type": "audio",
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
    """task-31277 (critique #4 P2): eight rows of chrome before the first
    content line. The identity line restates what the Media list already
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
    section header directly beneath it says the same word twice and costs
    four rows (bold text + top rule + padding + margin)."""
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
        title = screen.query_one("#library-media-viewer-analysis-title")
        assert title.display is False, title.display
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
