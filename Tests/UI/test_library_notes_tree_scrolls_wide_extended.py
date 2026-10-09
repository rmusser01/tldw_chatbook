"""TASK-34000.7, the slower variants of ``test_library_notes_tree_scrolls_wide``.

Deliberately NOT in ``scripts/ui_pr_gate_census.txt``: extra sizes, wheel
events, a breakpoint round trip and the Trash opener each cost a boot, and
the census file has to stay under 35 s. Same harness, same fixture.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from textual import events
from textual.widget import Widget
from textual.widgets import Button

from Tests.UI.test_library_notes_tree_scrolls_wide import (
    COMPACT_SIZES,
    DOWN_PRESSES,
    _assert_never_scrolls_horizontally,
    _many_notes,
    _open_notes_tree,
    _rows,
)
from Tests.UI.test_library_shell import (
    LibraryHarness,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_notes_compact,
    _wait_for_selector,
)


def _wheel(widget: Widget, delta_y: int) -> events.MouseScrollDown | events.MouseScrollUp:
    cls = events.MouseScrollDown if delta_y > 0 else events.MouseScrollUp
    return cls(widget, 0, 0, 0, delta_y, 0, False, False, False)


def _scrolled_ancestors(widget: Widget) -> list[tuple[str | None, float]]:
    """``(id, scroll_y)`` for every ancestor that is scrolled off its top."""
    return [
        (ancestor.id, ancestor.scroll_y)
        for ancestor in widget.ancestors
        if isinstance(ancestor, Widget) and ancestor.scroll_y > 0
    ]


async def _load_more_pages(pilot, screen, pages: int):
    """Press the root "More notes" pager ``pages`` times (20 rows each).

    The tall sizes (200x50, 235x52, 100x50) fit the first 22-row page, so
    they need the real pager -- TASK-18917's control, reachable only now
    that the list scrolls -- to overflow and exercise scrolling.

    Returns the LIVE ``#library-notes-list``: a page load replaces the list
    widget (a targeted canvas sync re-mounts ``#library-notes-list``, see
    backlog/docs/lessons-textual.md), so a handle taken before the press
    is a detached node with no children afterwards.
    """
    lst = screen.query_one("#library-notes-list")
    for _ in range(pages):
        before = len(_rows(screen))
        lst.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        pager = lst.children[-1]
        assert pager.has_class("library-notes-tree-pager"), pager.id
        assert lst.region.contains_region(pager.region), (
            f"the pager {pager.id} is not reachable: {pager.region} vs {lst.region}"
        )
        pager.press()
        await _wait_for_condition(
            pilot,
            lambda: len(screen.query(".library-notes-row")) >= before + 20,
            message="The next page of note rows never mounted.",
        )
        await pilot.pause()
        await pilot.pause()
        lst = screen.query_one("#library-notes-list")
    return lst


async def _walk_down(pilot, screen, presses: int) -> Widget:
    _rows(screen)[0].focus()
    await pilot.pause()
    for _ in range(presses):
        await pilot.press("down")
    await pilot.pause()
    await pilot.pause()
    focused = screen.focused
    assert focused is not None and focused.has_class("library-notes-row")
    return focused


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(200, 50), (235, 52)], ids=lambda s: f"{s[0]}x{s[1]}")
async def test_wide_sizes_with_the_rail_open_scroll_and_reveal(size) -> None:
    """AC#1's third size (235x52) and AC#5's 200x50 with the Nav rail open."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)
        shell = screen.query_one(".library-notes-route")
        assert shell.effective_layout.library_open, "the Nav rail should be open here"
        assert shell.library.display

        # The first 22-row page fits a 50-row terminal: with nothing to
        # scroll Textual shows no scrollbar and ``allow_vertical_scroll`` is
        # False by design. Load two more pages so the tree overflows.
        assert lst.styles.overflow_y == "auto"
        lst = await _load_more_pages(pilot, screen, 2)
        assert len(_rows(screen)) >= 60
        assert lst.max_scroll_y > 0, f"{size}: 60+ rows still fit {lst.region}"
        assert lst.allow_vertical_scroll is True
        _assert_never_scrolls_horizontally(size, screen, lst)
        lst.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert lst.scroll_y > 0
        last = lst.children[-1]
        assert lst.region.contains_region(last.region)

        focused = await _walk_down(pilot, screen, DOWN_PRESSES)
        assert lst.region.contains_region(focused.region), (
            f"{size}: {focused.id} at {focused.region} outside {lst.region}"
        )
        assert _scrolled_ancestors(lst) == [], "only the list itself may scroll"


@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(120, 36), (160, 45)], ids=lambda s: f"{s[0]}x{s[1]}")
async def test_wheel_over_a_row_scrolls_the_list(size) -> None:
    """AC#1: the wheel over the tree moves it (base: byte-identical frames).

    The event is posted to the ROW under the pointer: a Button owns no
    scrolling, so the event bubbles to the list, exactly as a real wheel
    does. Ten down then ten up returns to the top.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)
        row = _rows(screen)[1]
        await pilot.hover(f"#{row.id}")
        for _ in range(10):
            row.post_message(_wheel(row, 1))
        await pilot.pause()
        await pilot.pause()
        # Textual's pointer step is more than one row per event, so pin the
        # direction and the bounds, not a per-event distance.
        assert lst.scroll_y > 0, f"{size}: ten wheel-down events moved nothing"
        assert lst.scroll_y <= lst.max_scroll_y
        assert _scrolled_ancestors(lst) == []

        for _ in range(10):
            row.post_message(_wheel(row, -1))
        await pilot.pause()
        await pilot.pause()
        assert lst.scroll_y == 0


@pytest.mark.asyncio
async def test_one_down_past_the_edge_reveals_exactly_one_row() -> None:
    """AC#2: stepping past the bottom edge scrolls by ONE row, not half a pane.

    Textual's own ``focus()`` path centres the newly focused widget
    (``Screen.set_focus`` -> ``scroll_to_center``), which makes a Down walk
    jump half a pane at every edge. ``_move_library_list_row_focus`` asks
    for the minimal reveal instead, so the list behaves like every other
    row list (ListView, OptionList): one press, one row.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(160, 45)) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)
        assert lst.max_scroll_y > 0
        rows = _rows(screen)
        rows[0].focus()
        await pilot.pause()
        # Step to the last row that is fully inside the pane without any
        # scrolling having happened yet.
        guard = 0
        while True:
            focused = screen.focused
            index = rows.index(focused)
            nxt = rows[index + 1]
            if nxt.region.bottom > lst.region.bottom:
                break
            await pilot.press("down")
            await pilot.pause()
            guard += 1
            assert guard < 60
        assert lst.scroll_y == 0, "no scroll should have happened inside the pane"
        edge_row = screen.focused
        assert edge_row.region.bottom == lst.region.bottom

        await pilot.press("down")
        await pilot.pause()
        await pilot.pause()
        focused = screen.focused
        assert focused is rows[rows.index(edge_row) + 1]
        assert lst.scroll_y == 1, (
            f"one Down past the edge scrolled by {lst.scroll_y} rows "
            f"(list {lst.region}, focused {focused.region})"
        )
        assert focused.region.bottom == lst.region.bottom
        assert lst.region.contains_region(focused.region)

        # ...and Up past the top edge likewise reveals one row.
        lst.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        top_visible = next(
            row for row in rows if row.region.y >= lst.region.y
        )
        top_visible.focus(scroll_visible=False)
        await pilot.pause()
        before = lst.scroll_y
        await pilot.press("up")
        await pilot.pause()
        await pilot.pause()
        assert lst.scroll_y == before - 1
        assert screen.focused.region.y == lst.region.y


@pytest.mark.asyncio
async def test_breakpoint_round_trip_keeps_the_list_scrolling_without_recompose() -> None:
    """119 -> 120 -> 119 columns: the list is a scroll owner on both sides
    of ``LIBRARY_NOTES_COMPACT_BREAKPOINT`` and the canvas is not rebuilt
    (the identity pin from
    ``test_library_note_compact_labels_round_trip_without_recompose``)."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(119, 36)) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)
        await _wait_for_library_notes_compact(screen, pilot, True)
        # The compact pane at 119x36 is exactly 22 rows tall -- the first
        # page fits to the row -- so load one more page before the trip.
        lst = await _load_more_pages(pilot, screen, 1)
        canvas = screen.query_one("#library-notes-canvas")
        assert lst.max_scroll_y > 0, f"sanity: the tree fits at 119x36 {lst.region}"
        assert lst.allow_vertical_scroll is True

        await pilot.resize_terminal(120, 36)
        await _wait_for_library_notes_compact(screen, pilot, False)
        await pilot.pause()
        assert screen.query_one("#library-notes-canvas") is canvas
        lst = screen.query_one("#library-notes-list")
        assert lst.max_scroll_y > 0
        assert lst.allow_vertical_scroll is True
        lst.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert lst.scroll_y > 0
        _assert_never_scrolls_horizontally((120, 36), screen, lst)

        await pilot.resize_terminal(119, 36)
        await _wait_for_library_notes_compact(screen, pilot, True)
        await pilot.pause()
        assert screen.query_one("#library-notes-canvas") is canvas
        lst = screen.query_one("#library-notes-list")
        assert lst.max_scroll_y > 0
        assert lst.allow_vertical_scroll is True
        _assert_never_scrolls_horizontally((119, 36), screen, lst)


@pytest.mark.asyncio
async def test_recently_deleted_opener_is_reachable_by_keyboard_alone() -> None:
    """AC#1: the Trash opener is the list's last child; Tab from the last
    row lands on it, reveals it, and Enter opens the Trash view."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    app.notes_scope_service.list_deleted_notes = AsyncMock(
        return_value={
            "items": [
                {
                    "id": "trash-1",
                    "title": "Old draft",
                    "version": 2,
                    "last_modified": "2026-07-01T00:00:00+00:00",
                }
            ],
            "total": 1,
        }
    )
    host = LibraryHarness(app)

    async with host.run_test(size=(120, 36)) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)
        opener = await _wait_for_selector(screen, pilot, "#library-notes-trash-open")
        assert opener.parent is lst and lst.children[-1] is opener
        assert not lst.region.contains_region(opener.region), (
            "sanity: the opener should start below the fold at 120x36"
        )

        _rows(screen)[-1].focus()
        await pilot.pause()
        await pilot.pause()
        # The root pager sits between the last row and the opener, so the
        # opener is at most the second Tab stop from there.
        for _ in range(3):
            await pilot.press("tab")
            await pilot.pause()
            if screen.focused is opener:
                break
        await pilot.pause()
        assert screen.focused is opener, getattr(screen.focused, "id", None)
        assert lst.region.contains_region(opener.region), (
            f"Tab focused the opener but left it at {opener.region} "
            f"(list {lst.region}, scroll_y={lst.scroll_y})"
        )

        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-notes-trash-header")


@pytest.mark.asyncio
async def test_compact_tall_layout_has_exactly_one_scroll_owner() -> None:
    """Review focus: nothing double-scrolls in the compact layout. At
    100x50 the compact rule applies AND the pane is tall; the list is the
    only scroll owner that moves, and a Down walk never scrolls an
    ancestor."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=(100, 50)) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)
        await _wait_for_library_notes_compact(screen, pilot, True)
        # The first page fits a 50-row pane; load two more so it overflows
        # (see ``test_wide_sizes_with_the_rail_open_scroll_and_reveal``).
        lst = await _load_more_pages(pilot, screen, 2)
        assert lst.max_scroll_y > 0
        assert lst.allow_vertical_scroll is True
        _assert_never_scrolls_horizontally((100, 50), screen, lst)

        focused = await _walk_down(pilot, screen, DOWN_PRESSES)
        assert lst.region.contains_region(focused.region)
        assert _scrolled_ancestors(lst) == [], (
            f"an ancestor scrolled too: {_scrolled_ancestors(lst)}"
        )

        lst.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert lst.scroll_y == lst.max_scroll_y
        assert _scrolled_ancestors(lst) == []
        last = lst.children[-1]
        assert lst.region.contains_region(last.region)


@pytest.mark.asyncio
@pytest.mark.parametrize("size", COMPACT_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
async def test_compact_sizes_still_scroll_and_never_scroll_horizontally(size) -> None:
    """AC#3: the compact sizes keep their scrolling (a regression pin --
    green on the base) and never scroll horizontally. The wide sizes get the
    same horizontal pin inside ``test_wide_notes_list_is_a_scroll_owner``."""
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)
        assert screen._notes_state.compact is True
        assert lst.allow_vertical_scroll is True
        lst.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert lst.scroll_y > 0
        _assert_never_scrolls_horizontally(size, screen, lst)
