"""Library ▸ Notes wave-4 group `layout`: toolbar width, New note, landing.

Covers tasks 32544 (the folder-actions row clips to "Remove pl" beside an
open note, and the New note view is squeezed beside an empty list), 32547
(60x24 "New" never promotes the New note view), 32557 ("Add from files…"
painted "Add from" against the grip after a resize), 32546 (the landing's
"From your Library" rows carry no focus shape) and 32549 (three disabled
controls state no reason on screen).

Every width here is resolved by the production resolver rather than typed,
and every label assertion reads what production composed.
"""

from __future__ import annotations

import dataclasses

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_notes_state import LibraryNotesListRow
from tldw_chatbook.Library.library_notes_tree_state import (
    LibraryNotesTreeProjection,
    LibraryNotesTreeRow,
)
from tldw_chatbook.Library.library_shell_state import (
    LIBRARY_ROW_BROWSE_NOTES,
    LIBRARY_ROW_CREATE_NOTE,
)
from tldw_chatbook.Notes.note_folder_models import FolderPlacementId
from tldw_chatbook.UI.Library_Modules.screen_constants import (
    LIBRARY_NOTES_READER_PROFILE,
)
from tldw_chatbook.UI.Screens.library_screen import LibraryScreen
from tldw_chatbook.Utils.adaptive_reader_state import (
    AdaptiveReaderLayoutPreferences,
    resolve_adaptive_reader_layout,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import LibraryNotesCanvas

from Tests.UI.test_library_notes_wave_list import (
    _CanvasApp,
    _layout_screen_fake,
    _list_state,
    assert_every_action_fits,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _two_notes,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

#: The critique's wide geometry and its compact one.
WIDE = (235, 52)
COMPACT = (60, 24)

#: The house focus bar (task-31983 / task-32359): focus is a SHAPE here.
THICK_LEFT_GLYPH = "█"


def _painted_rows(app) -> list[str]:
    """What the compositor actually put on screen, row by row."""
    return [
        "".join(segment.text for segment in strip)
        for strip in app.screen._compositor.render_strips()
    ]


def _items_width(terminal_width: int, *, reader_has_item: bool) -> int:
    """What the production resolver gives the Notes list at this width."""
    return resolve_adaptive_reader_layout(
        terminal_width,
        AdaptiveReaderLayoutPreferences(),
        LIBRARY_NOTES_READER_PROFILE,
        reader_has_item=reader_has_item,
    ).items_width


def _select_mode_state():
    """The select strip with rows rendered and none of them checked."""
    rows = tuple(
        LibraryNotesListRow(note_id=f"n{index}", title=f"Note {index}", age_label="2m")
        for index in range(1, 4)
    )
    return dataclasses.replace(
        _list_state(rows=rows),
        select_mode=True,
        selected_count=0,
        result_count=len(rows),
    )


def _note_selected_projection():
    """A tree with the open note's placement selected.

    The toolbar pins that exist all select a FOLDER, whose actions are
    "New folder / Rename / Move / Remove" (46 cells). With a NOTE selected
    the row is "New folder / Add to folder / Move note / Remove placement"
    (62 cells) -- the frame the critique read as "Remove pl".
    """
    return LibraryNotesTreeProjection(
        rows=(
            LibraryNotesTreeRow(
                placement_id=FolderPlacementId.folder("work"),
                kind="folder",
                label="Work",
                depth=0,
                folder_id="work",
                breadcrumb="Work",
                expanded=True,
            ),
            LibraryNotesTreeRow(
                placement_id=FolderPlacementId.note("work", "n1", "m1"),
                kind="note",
                label="Quarterly plan",
                depth=1,
                note_id="n1",
                folder_id="work",
                membership_id="m1",
                breadcrumb="Work / Quarterly plan",
            ),
        )
    )


def _note_selected_app(pane_width: int, **overrides) -> _CanvasApp:
    """The list toolbar with a note's placement selected, at one pane width."""
    kwargs = dict(
        list_state=_list_state(),
        tree_projection=_note_selected_projection(),
        tree_selected_placement_id=FolderPlacementId.note("work", "n1", "m1"),
        import_receipt_available=True,
        lasting_sync_snapshot=None,
    )
    kwargs.update(overrides)
    return _CanvasApp(pane_width=pane_width, **kwargs)


# -- task-32544 AC#1: the folder-actions row beside an open note ----------


@pytest.mark.asyncio
async def test_notes_toolbar_paints_whole_labels_with_a_note_open_at_235x52() -> None:
    """task-32544 AC#1/AC#3: no toolbar label is cut off beside an open note.

    The width is the one the production resolver hands the list at 235
    columns with a note open, not a typed constant; live at 235x52 that
    pane painted "New folder  Add to folder  Move note  Remove pl".
    """
    pane_width = _items_width(WIDE[0], reader_has_item=True)
    app = _note_selected_app(pane_width)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        assert_every_action_fits(app)
        labels = {
            str(button.label)
            for button in app.query(".library-canvas-action")
        }
        assert "Remove placement" in labels


@pytest.mark.asyncio
async def test_notes_folder_actions_wrap_rather_than_run_off_the_pane() -> None:
    """task-32544 AC#1: the row that cannot fit takes another row.

    The guide's own narrow-pane rule ("the toolbar moves the action that
    does not fit onto a row of its own"), applied to the folder actions.
    """
    pane_width = _items_width(WIDE[0], reader_has_item=True)
    app = _note_selected_app(pane_width)
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        actions = [
            widget
            for widget in app.query(".library-canvas-action")
            if widget.id
            and widget.id.startswith(
                ("library-notes-folder-", "library-notes-placement-")
            )
        ]
        assert len(actions) == 4, [widget.id for widget in actions]
        assert len({widget.region.y for widget in actions}) >= 2


# -- task-32544 AC#2: the New note view's share of the canvas -------------


def test_the_new_note_view_gets_the_width_its_status_needs_at_235x52() -> None:
    """task-32544 AC#2: the empty list does not keep half the canvas.

    The create view's identity lives in ``_library_selected_row_id``, not
    in ``_notes_state.view`` (which stays "list"), so the resolver was
    told the work pane was empty and handed the list 138 of 235 columns
    while "Ready · Next: Press Blank note, or choose a template." wrapped
    inside 48.
    """
    fake, shell = _layout_screen_fake(width=WIDE[0], view="list")
    fake._library_selected_row_id = LIBRARY_ROW_CREATE_NOTE

    LibraryScreen._sync_library_notes_reader_layout_from_shell(fake)

    assert shell.applied is not None
    assert shell.applied.reader_width >= 60, (
        f"the New note view got {shell.applied.reader_width} columns beside "
        f"a {shell.applied.items_width}-column empty list"
    )
    assert shell.applied.items_width <= WIDE[0] // 2


def test_the_notes_list_still_keeps_the_freed_width_with_nothing_open() -> None:
    """The create fix must not take the empty list's width away (task-32127).

    Negative control for the case above: on the list itself, with no note
    and no create view, an empty work pane still hands its columns over.
    """
    fake, shell = _layout_screen_fake(width=WIDE[0], view="list")
    fake._library_selected_row_id = LIBRARY_ROW_BROWSE_NOTES

    LibraryScreen._sync_library_notes_reader_layout_from_shell(fake)

    assert shell.applied is not None
    assert shell.applied.items_width >= 100


# -- task-32547: 60x24 "New" must promote the New note view --------------


def test_new_at_sixty_columns_gives_the_create_view_the_stage() -> None:
    """task-32547 AC#1 at the resolver: the list stops owning the stage."""
    fake, shell = _layout_screen_fake(width=COMPACT[0], view="list")
    fake._library_selected_row_id = LIBRARY_ROW_CREATE_NOTE

    LibraryScreen._sync_library_notes_reader_layout_from_shell(fake)

    assert shell.applied is not None
    assert shell.applied.items_open is False, (
        f"the list still holds {shell.applied.items_width} of "
        f"{COMPACT[0]} columns while the New note view is the task in hand"
    )
    assert shell.applied.reader_width >= 40


@pytest.mark.asyncio
async def test_new_at_60x24_promotes_the_create_view_with_blank_note_focused() -> None:
    """task-32547 AC#1/AC#2/AC#3 through the real screen at 60x24."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=COMPACT) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-new")

        screen.query_one("#library-notes-new", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-create-blank")
        blank = screen.query_one("#library-notes-create-blank", Button)
        assert blank.display
        assert blank.region.width > 0 and blank.region.height > 0
        assert screen.focused is blank
        assert screen._library_focus_enter_label(blank) == "create note"
        # task-32546's footer clause is not landing-only (review M7): it
        # appends the chip on ANY Library surface whose static set has no
        # "enter" to replace, and this route is a second such surface. Pin
        # it here so the screen-wide behaviour is pinned somewhere other
        # than the landing it was written for.
        assert ("enter", "create note") in (
            screen._library_footer_shortcuts_for_current_state()
        )
        assert not any(
            key == "enter"
            for key, _ in screen._library_route_shortcuts_for_current_state()
        ), "this route gained a static enter chip; the clause under test cannot fire"

        await pilot.press("escape")
        await _wait_for_selector(screen, pilot, "#library-notes-new")
        assert not screen.query("#library-notes-create-blank")


# -- task-32557: the toolbar's width after a resize ----------------------


@pytest.mark.asyncio
async def test_notes_toolbar_labels_paint_whole_or_elided_at_sixty_columns() -> None:
    """task-32557 AC#1/AC#2: a narrowed pane re-shapes before it paints.

    Driven through the real screen and a real resize, because that is where
    the defect lived: the canvas learned its pane width only from the next
    state sync, so live, narrowing a merged 235-column list to 60 painted
    "New  Sort: Newest  Select  Add from   s" -- the merged row still in a
    50-column pane.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        assert screen.query("#library-notes-action-rows"), "not merged at 235"
        # One ordinary state sync at the wide size, which is what stamps the
        # screen's contract width onto the canvas -- and what made the live
        # walk's toolbar keep 138 cells' worth of shape at 50 cells. Without
        # it this canvas has never been told a width at all and falls back
        # to measuring itself, which is not the state the defect lives in.
        screen.query_one("#library-notes-select-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-select-all")
        screen.query_one("#library-notes-select-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        await _wait_for_condition(
            pilot,
            lambda: canvas.pane_width > COMPACT[0],
            message="the wide pane width never reached the canvas",
        )

        # No state sync is provoked here on purpose: in the live app
        # nothing syncs this canvas after a resize, so the merged row stayed
        # on screen until an unrelated interaction. Settle the frames the
        # resize itself produces and read what is painted.
        await pilot.resize_terminal(*COMPACT)
        for _ in range(3):
            await pilot.pause()

        # The exact shipped spelling, not `endswith(("files…", "…"))`:
        # the second alternative subsumed the first and the pair passed on
        # almost any string (review M6).
        add_from = screen.query_one("#library-notes-add-from-files", Button)
        assert str(add_from.label) == "Add from files…", str(add_from.label)
        offenders = [
            (widget.id, widget.region)
            for widget in screen.query(".library-canvas-action")
            if widget.region.width
            and widget.region.right > canvas.region.right
        ]
        assert not offenders, (
            f"actions painted off the {canvas.region.width}-column canvas: "
            f"{offenders}"
        )


@pytest.mark.asyncio
async def test_a_pane_that_widens_again_records_the_width_it_was_given() -> None:
    """task-32557, review M1: growth does not re-shape, but it is recorded.

    `apply_pane_width` deliberately re-shapes only on a shrink -- re-shaping
    on growth costs the in-place breakpoint path its widget identity. It
    used to return on growth without recording the width either, and
    `_effective_pane_width` gives `pane_width` priority over the measured
    width, so 235 -> 60 -> 235 left the canvas composing the 50-cell shape
    into a 138-cell pane until an unrelated state sync re-stamped it. Dev
    does not have that state: dev's `pane_width` never moves off its
    compose value, so dev ends this round trip correctly wide.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        screen.query_one("#library-row-browse-notes").press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        canvas = screen.query_one("#library-notes-canvas", LibraryNotesCanvas)
        # One ordinary state sync stamps the screen's contract width.
        screen.query_one("#library-notes-select-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-select-all")
        screen.query_one("#library-notes-select-toggle", Button).press()
        await _wait_for_selector(screen, pilot, "#library-notes-add-from-files")
        await _wait_for_condition(
            pilot,
            lambda: canvas.pane_width > COMPACT[0],
            message="the wide pane width never reached the canvas",
        )
        # Production's own stamped contract, not a number typed here: the
        # shell hands the canvas the width the reader layout resolved for
        # the pane INSIDE the rail, which is narrower than the resolver's
        # answer for the whole terminal.
        wide_pane = canvas.pane_width

        await pilot.resize_terminal(*COMPACT)
        for _ in range(3):
            await pilot.pause()
        narrow_pane = canvas.pane_width
        assert narrow_pane < wide_pane, "the shrink never reached the canvas"

        # Back to the wide terminal, with NO state sync in between -- which
        # is the live sequence: nothing syncs this canvas after a resize.
        await pilot.resize_terminal(*WIDE)
        for _ in range(3):
            await pilot.pause()

        assert canvas.pane_width == wide_pane, (
            f"the widened pane was dropped: canvas still holds {canvas.pane_width} "
            f"after {wide_pane} -> {narrow_pane} -> {wide_pane}"
        )
        assert canvas._effective_pane_width() == wide_pane


# -- task-32546: the landing's "From your Library" rows ------------------


async def _landing_recent(host, pilot):
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    await _wait_for_selector(screen, pilot, ".library-hub-recent")
    return screen, screen.query(".library-hub-recent").first(Button)


@pytest.mark.asyncio
async def test_landing_recent_rows_show_the_focus_shape_and_the_footer_names_them() -> None:
    """task-32546 AC#1/AC#3: shape-based focus, and a footer that names it.

    Live, twelve Tabs and an F6 toggle changed nothing but the row's own
    background (rgb(30,30,30) -> rgb(28,70,102)) -- colour only, the exact
    thing task-32359 removed from the rail rows one pane away.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE) as pilot:
        screen, recent = await _landing_recent(host, pilot)
        recent.focus()
        await pilot.pause()

        assert recent.styles.border_left[0], (
            f"{recent.id} has no focus shape: {recent.styles.border_left!r}"
        )
        line = _painted_rows(host.app)[recent.region.y]
        assert line[recent.region.x] == THICK_LEFT_GLYPH, line
        assert screen._library_focus_enter_label(recent) == "open notes"
        chips = dict(screen._library_footer_shortcuts_for_current_state())
        assert chips.get("enter") == "open notes", chips


@pytest.mark.asyncio
async def test_f6_on_the_landing_lands_on_a_marked_target() -> None:
    """task-32546 AC#2: F6's landing target carries the same focus shape."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_two_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _wait_for_selector(screen, pilot, ".library-hub-recent")

        for _ in range(4):
            screen.action_focus_next_workbench_pane()
            await pilot.pause()
            focused = screen.focused
            if focused is not None and focused.has_class("library-hub-recent"):
                break
        else:  # pragma: no cover - the landing always offers the canvas pane
            pytest.fail("F6 never reached the landing's canvas target")

        assert focused.styles.border_left[0], (
            f"F6's target {focused.id} is unmarked: "
            f"{focused.styles.border_left!r}"
        )


# -- task-32549: three disabled controls state their reason --------------


@pytest.mark.asyncio
async def test_sort_states_its_reason_while_a_filter_is_showing() -> None:
    """task-32549 AC#2: "○ Sort: Newest" said nothing about the filter.

    On the shared line rather than in the label, and measured that way:
    the label spelling painted "○ Sort unavailable — clear the" against the
    grip on the 42-column pane a 100x30 terminal gives this list.
    """
    pane_width = _items_width(WIDE[0], reader_has_item=False)
    app = _CanvasApp(
        pane_width=pane_width,
        list_state=_list_state(),
        filter_value="list",
        tree_projection=_note_selected_projection(),
        import_receipt_available=True,
    )
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        sort = app.query_one("#library-notes-sort", Button)
        assert sort.disabled
        reason = app.query_one("#library-notes-sort-disabled-reason", Static)
        assert str(reason.renderable) == "Sort unavailable — clear the filter"
        assert_every_action_fits(app)


@pytest.mark.asyncio
async def test_export_selected_states_its_reason_with_nothing_selected() -> None:
    """task-32549 AC#2: the select strip's blocked action says why.

    Its own row has no cells to spare -- task-32261 already had to hide the
    in-row counter to keep this action on the pane -- and a line of its own
    costs the tree a row at 60x20, which
    ``test_library_note_60x20_navigator_state_allocation`` caught when this
    was first written that way. The line under the strip already exists and
    already carries the count, so it names the blocked action too.
    """
    pane_width = _items_width(WIDE[0], reader_has_item=False)
    app = _CanvasApp(
        pane_width=pane_width,
        list_state=_select_mode_state(),
        # A filter that WOULD block Sort, so the absence asserted below is
        # the select-mode guard and not a vacuous query.
        filter_value="list",
        tree_projection=_note_selected_projection(),
        import_receipt_available=True,
    )
    async with app.run_test(size=WIDE) as pilot:
        await pilot.pause()
        export = app.query_one("#library-notes-export-selected", Button)
        assert export.disabled
        reason = app.query_one("#library-notes-selection-status", Static)
        assert str(reason.renderable) == (
            "0 selected — Export selected unavailable"
        )
        # Select mode replaces the whole toolbar, so the Sort control is not
        # on screen and neither is its reason.
        assert not app.query("#library-notes-sort-disabled-reason")
        assert_every_action_fits(app)


@pytest.mark.asyncio
async def test_resolution_history_states_its_reason_when_it_is_unreachable() -> None:
    """task-32549 AC#2: the review's blocked history opener says why.

    Composed through the real Add-from-files canvas in its review phase
    with a source that cannot reach history, and read off the mounted
    widgets -- the pinned action bar is one ``max-height: 3`` row, so the
    reason takes the shared line the canvas already gives "○ Server notes".
    """
    from Tests.Widgets.Library.test_library_notes_add_from_files_canvas import (
        _Host,
        _conflict_review,
    )
    from tldw_chatbook.Library.library_notes_lasting_sync_state import (
        initial_lasting_sync_snapshot,
    )

    snapshot = dataclasses.replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="review",
        review=dataclasses.replace(_conflict_review(), source="setup"),
    )
    app = _Host(snapshot)
    async with app.run_test(size=COMPACT) as pilot:
        await pilot.pause()
        history = app.query_one("#notes-sync-history-open", Button)
        assert history.disabled
        assert str(history.label) == "○ Resolution history"
        reason = app.query_one("#notes-sync-history-disabled-reason", Static)
        assert str(reason.renderable) == (
            "Resolution history unavailable — it starts after this root is activated"
        )


@pytest.mark.asyncio
async def test_resolution_history_reason_clears_when_activation_lands_in_place() -> (
    None
):
    """task-32610: activation used to leave this line stuck.

    ``_sync_review`` -- the in-place update ``sync_state`` takes when a
    review snapshot changes without its root_id/token/stale/page/rows/
    receipts changing -- updated the button's own label but never the
    separate visible reason line, so the pane still read "unavailable"
    after the root the reason names had already been activated.
    """
    from Tests.Widgets.Library.test_library_notes_add_from_files_canvas import (
        _Host,
        _conflict_review,
    )
    from tldw_chatbook.Library.library_notes_lasting_sync_state import (
        initial_lasting_sync_snapshot,
    )

    snapshot = dataclasses.replace(
        initial_lasting_sync_snapshot(lasting_available=True),
        phase="review",
        review=dataclasses.replace(_conflict_review(), source="setup"),
    )
    app = _Host(snapshot)
    async with app.run_test(size=COMPACT) as pilot:
        await pilot.pause()
        reason = app.query_one("#notes-sync-history-disabled-reason", Static)
        assert reason.display is True

        canvas = app.query_one("LibraryNotesAddFromFilesCanvas")
        canvas.sync_state(
            dataclasses.replace(
                snapshot,
                review=dataclasses.replace(snapshot.review, source="root"),
            )
        )
        await pilot.pause()

        history = app.query_one("#notes-sync-history-open", Button)
        assert not history.disabled
        assert str(history.label) == "Resolution history"
        assert reason.display is False


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_width", [235, 100, 60])
async def test_the_blocked_sort_reason_fits_every_pane_the_list_is_given(
    terminal_width,
) -> None:
    """task-32549 AC#2 at the three widths the critique ran at.

    The label spelling of this reason clipped at 100x30 -- the pane there is
    42 cells and "○ Sort unavailable — clear the filter" needs 41 beside
    "New" -- which is why it lives on the shared line.
    """
    pane_width = _items_width(terminal_width, reader_has_item=False)
    app = _CanvasApp(
        pane_width=pane_width,
        list_state=_list_state(),
        filter_value="list",
        tree_projection=_note_selected_projection(),
        import_receipt_available=True,
    )
    async with app.run_test(size=(terminal_width, 30)) as pilot:
        await pilot.pause()
        reason = app.query_one("#library-notes-sort-disabled-reason", Static)
        assert str(reason.renderable) == "Sort unavailable — clear the filter"
        # Against the RENDERED canvas, like `assert_every_action_fits` does,
        # rather than against the number this test handed in (review M6).
        canvas = app.query_one("#library-notes-canvas", LibraryNotesCanvas)
        assert reason.region.right <= canvas.region.right
        assert_every_action_fits(app)
