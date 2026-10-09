"""TASK-34000.8, the slower arms: the Library Notes editor header at the
remaining sizes the task names, a narrower editor pane at the wide sizes,
the long title, a short terminal, the 235 -> 120 -> 235 round trip and the
delete prompt while the header is stacked.

Not in the PR-lane census (``scripts/ui_pr_gate_census.txt``); the gated
arms are in ``test_library_note_header_fits_pane.py``. Same harness: the
real ``LibraryScreen`` under ``LibraryHarness`` with the app bundle.

"Rail open": the live shell keeps the Navigation rail open at 160x45 and
200x50, which narrows the editor pane; the harness closes the rail while a
note is being worked on (the work-session override), so these arms narrow
the pane the other way the shell allows -- a wider Items pane through the
``notes_reader`` preference -- and assert the same promise: every header
control whole inside whatever pane the shell resolved.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static, TextArea

from Tests.UI.test_library_note_header_fits_pane import (
    BODY,
    DISCARD,
    HEADER_CONTROLS,
    PANE,
    SAVE,
    STATUS,
    USE_IN_CONSOLE,
    _assert_status_is_a_readable_word,
    _assert_whole_inside_pane,
    _is_on_screen,
    _new_blank_note,
    _open_first_note,
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
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    _HEADER_ONE_ROW_MIN_WIDTH,
    _header_shape,
)

pytestmark = pytest.mark.bootstrap_profile

SECOND_ROW = "#library-note-header-second-row"
PRIMARY = "#library-note-primary-actions"


def _layout_name(widget) -> str:
    """``styles.layout`` is a Layout object whose ``str`` is ``<vertical>``."""
    return widget.styles.layout.name

#: AC#1's remaining sizes, each with the shell's default panes and with a
#: wider Items pane that leaves the editor less than one header row needs.
EXTENDED_SIZES = [(140, 40), (200, 50), (235, 52), (200, 24)]
ITEMS_WIDTH_OVERRIDES = [None, 120]

LONG_TITLE = "Advisor meeting 2026-09-24 — " + "a very long title that keeps going " * 6


def _host(*, items_width: int | None = None, notes=None) -> LibraryHarness:
    app = _build_test_app()
    if items_width is not None:
        app.app_config.setdefault("library", {})["notes_reader"] = {
            "items_open": True,
            "items_width": items_width,
        }
    _seed_conversations(app, _two_conversations(), notes=notes or _two_notes())
    return LibraryHarness(app)


def _assert_header_whole(size, host, screen) -> None:
    for selector, label in (
        ("#library-note-edit", "Edit"),
        ("#library-note-preview", "Preview"),
        ("#library-note-context", "Info"),
        (SAVE, "Save"),
        (USE_IN_CONSOLE, "Use in Console"),
    ):
        _assert_whole_inside_pane(size, host, screen, selector, label)
    _assert_status_is_a_readable_word(size, screen)


def _shape_matches_the_pane(screen) -> None:
    """The rendered shape is the one ``_header_shape`` picks for the pane."""
    pane = screen.query_one(PANE)
    second_row = screen.query_one(SECOND_ROW)
    expected = _header_shape(pane.region.width, screen._notes_state.compact)
    assert (_layout_name(second_row) == "vertical") == (
        expected or screen._notes_state.compact
    ), (
        f"pane {pane.region.width} wide, one-row minimum "
        f"{_HEADER_ONE_ROW_MIN_WIDTH}, layout {second_row.styles.layout}"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("items_width", ITEMS_WIDTH_OVERRIDES, ids=lambda w: f"items{w}")
@pytest.mark.parametrize("size", EXTENDED_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
async def test_every_header_control_is_whole_at_the_wide_sizes(size, items_width) -> None:
    """AC#1 / AC#4 at 140x40, 200x50, 235x52 and a 24-row 200-column
    terminal, with the shell's own panes and with the editor narrowed by a
    wider Items pane; the shape rendered is the one the pane's width
    picks."""
    host = _host(items_width=items_width)
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note(host, pilot)
        _assert_header_whole(size, host, screen)
        _shape_matches_the_pane(screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 36), (235, 52)], ids=lambda s: f"{s[0]}x{s[1]}")
async def test_a_very_long_title_never_pushes_a_header_control_off_the_pane(size) -> None:
    """Review focus: the heading row holds the title; the header's second
    row must not inherit its width."""
    notes = [dict(_two_notes()[0], title=LONG_TITLE), _two_notes()[1]]
    host = _host(notes=notes)
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note(host, pilot)
        assert screen._library_note_session.snapshot.title == LONG_TITLE
        _assert_header_whole(size, host, screen)
        pane = screen.query_one(PANE)
        assert pane.region.contains_region(screen.query_one(SECOND_ROW).region)


@pytest.mark.asyncio
async def test_round_trip_re_shapes_in_place_without_recomposing() -> None:
    """235 -> 120 -> 235: the header stacks and unstacks on the SAME
    widgets (no recompose -- the in-place path task-32557 protects), with
    every control whole at each stop."""
    host = _host()
    async with host.run_test(size=(235, 52)) as pilot:
        screen = await _open_first_note(host, pilot)
        before = {
            selector: screen.query_one(selector)
            for selector in HEADER_CONTROLS + (SECOND_ROW, PRIMARY, STATUS)
        }
        _shape_matches_the_pane(screen)
        assert _layout_name(before[SECOND_ROW]) == "horizontal"

        for size, layout in (((120, 36), "vertical"), ((235, 52), "horizontal")):
            await pilot.resize_terminal(*size)
            await _wait_for_condition(
                pilot,
                lambda: screen.query_one("#library-shell-grid").region.width == size[0],
                message=f"{size}: the shell never took the new width.",
            )
            await pilot.pause()
            await pilot.pause()
            for selector, widget in before.items():
                assert screen.query_one(selector) is widget, (
                    f"{size}: {selector} was recomposed"
                )
            assert _layout_name(before[SECOND_ROW]) == layout, (
                f"{size}: layout {before[SECOND_ROW].styles.layout}"
            )
            _assert_header_whole(size, host, screen)
            _shape_matches_the_pane(screen)


@pytest.mark.asyncio
async def test_the_delete_prompt_hides_the_stacked_header_actions() -> None:
    """Task 2's contract (TASK-34000.13) with the header stacked: while the
    delete prompt is confirming, the two action rows are not displayed and
    the prompt's Cancel is the focused, visible control."""
    size = (120, 36)
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _open_first_note(host, pilot)
        assert _layout_name(screen.query_one(SECOND_ROW)) == "vertical"
        screen.query_one("#library-note-context", Button).press()
        await pilot.pause()
        await pilot.pause()
        screen.query_one("#library-note-context-delete", Button).press()
        await pilot.pause()
        await pilot.pause()
        assert screen._notes_state.confirming_delete is True
        assert screen.query_one(PRIMARY).display is False
        cancel = screen.query_one("#library-note-delete-cancel", Button)
        assert screen.focused is cancel
        assert _is_on_screen(screen, cancel)
        assert ("enter", "save note") not in screen._library_notes_footer_shortcuts()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("size", "expected_label"),
    [((120, 36), "Discard"), ((160, 45), "Discard new note")],
    ids=["120x36-short", "160x45-full"],
)
async def test_discard_wording_follows_the_task_rows_width(size, expected_label) -> None:
    """Review M-2 (fix round 1): the stacked task row keeps the full
    "Discard new note" wherever its own width holds it (160x45: 82+ cells
    against the 53 the row needs) and shortens to "Discard" only where it
    cannot (120x36: 48 cells). RED on 8b935e56e6 at 160x45, where the
    wording followed the shape and read "Discard"."""
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _new_blank_note(host, pilot)
        discard = screen.query_one(DISCARD, Button)
        await _wait_for_condition(
            pilot,
            lambda: discard.display and discard.visible,
            message="An untouched new note never offered Discard new note.",
        )
        assert _layout_name(screen.query_one(SECOND_ROW)) == "vertical"
        assert str(discard.label) == expected_label
        _assert_whole_inside_pane(size, host, screen, DISCARD, expected_label)
        _assert_whole_inside_pane(size, host, screen, USE_IN_CONSOLE, "Use in Console")


@pytest.mark.asyncio
async def test_new_note_discard_keeps_the_task_row_width_on_a_wide_stage() -> None:
    """The one-row shape at 235x52: Discard new note (wide wording) is whole,
    and hiding it keeps the task row's width and the mode row's x."""
    size = (235, 52)
    host = _host()
    async with host.run_test(size=size) as pilot:
        screen = await _new_blank_note(host, pilot)
        discard = screen.query_one(DISCARD, Button)
        await _wait_for_condition(
            pilot,
            lambda: discard.display and discard.visible,
            message="An untouched new note never offered Discard new note.",
        )
        assert _layout_name(screen.query_one(SECOND_ROW)) == "horizontal"
        assert str(discard.label) == "Discard new note"
        _assert_whole_inside_pane(size, host, screen, DISCARD, "Discard new note")
        task_actions = screen.query_one("#library-note-task-actions")
        width = task_actions.region.width
        screen.query_one(BODY, TextArea).text = "typed"
        await _wait_for_condition(
            pilot,
            lambda: not discard.visible,
            message="Discard new note never went away after typing.",
        )
        await pilot.pause()
        assert task_actions.region.width == width
        assert discard not in screen.focus_chain
        status = screen.query_one(STATUS, Static)
        assert status.region.width >= 12
