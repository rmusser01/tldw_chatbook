"""TASK-34000.7: the Library Notes tree scrolls at 120+ columns, and keyboard
focus never lands on an unseen row.

Review finding N-04 (qa/notes-library-ux-review-2026-10-02): at 120 columns
and wider ``#library-notes-list`` was a plain ``Vertical`` whose default
``overflow: hidden`` left it with no scroll owner -- only the compact rule
(below ``LIBRARY_NOTES_COMPACT_BREAKPOINT``) gave it ``overflow-y: auto``.
So the wheel changed nothing, and Down/Tab moved focus onto rows painted
below the pane's border (``focus()`` asks every ancestor to scroll the row
into view, and a ``hidden`` owner declines), so Enter opened a note the
user never saw.

Every test here mounts the real ``LibraryScreen`` under ``LibraryHarness``
(``CSS_PATH`` = the app bundle): without the bundle the wide and compact
rules are both absent and a ``region``/``scroll_y`` claim measures nothing
(backlog/docs/lessons-textual.md). 60 unfiled notes are seeded so the
tree pages at 20 rows and its pager is the list's last child.

On the base (origin/dev 0254a6bd34) the first two tests fail at both wide
sizes for the stated reason: ``allow_vertical_scroll`` is False, ``scroll_y``
stays 0.0 after ``scroll_end``, and the 30th focused row sits at y=48 on a
36-row screen (y=47 on a 45-row one). The slower variants (extra sizes,
wheel events, a breakpoint round trip, the Trash opener) live in
``test_library_notes_tree_scrolls_wide_extended.py``, outside the PR lane.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_library_shell import (
    LibraryHarness,
    _active_library_screen,
    _build_test_app,
    _seed_conversations,
    _two_conversations,
    _wait_for_condition,
    _wait_for_library_shell,
    _wait_for_selector,
)

#: Enough notes that the tree pages (20 rows per page) and the pager is the
#: 21st row -- so the list always has more rows than the pane at the sizes
#: the task names, and the LAST child is a control a user must reach.
NOTE_COUNT = 60

#: Down presses past the end of the first page (20 note rows + the folder
#: row): the focused row is the deepest one, well below the fold on the base.
DOWN_PRESSES = 30

WIDE_SIZES = [(120, 36), (160, 45)]
COMPACT_SIZES = [(80, 24), (100, 30)]


def _many_notes(count: int = NOTE_COUNT) -> list[dict]:
    """``count`` unfiled notes with distinct ages, newest first by index."""
    return [
        {
            "id": f"n-{index:03d}",
            "title": f"Note {index:03d}",
            "content": f"body {index}",
            "last_modified": f"2026-07-{(index % 28) + 1:02d}T12:00:00+00:00",
            "version": 1,
            "keywords": [],
        }
        for index in range(count)
    ]


async def _open_notes_tree(host, pilot):
    """Land on the Notes list and return ``(screen, list)`` once rows exist."""
    screen = _active_library_screen(host)
    await _wait_for_library_shell(screen, pilot)
    screen.query_one("#library-row-browse-notes").press()
    await _wait_for_selector(screen, pilot, ".library-notes-row")
    await _wait_for_condition(
        pilot,
        lambda: len(screen.query(".library-notes-row")) >= 20,
        message="The first page of note rows never mounted.",
    )
    await pilot.pause()
    return screen, screen.query_one("#library-notes-list")


def _rows(screen) -> list[Button]:
    return list(screen.query(".library-notes-row").results(Button))


@pytest.mark.asyncio
@pytest.mark.parametrize("size", WIDE_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
async def test_wide_notes_list_is_a_scroll_owner(size) -> None:
    """AC#1 / AC#4 (sharpened): the wide list owns its vertical scrolling.

    ``max_scroll_y > 0`` is kept as a sanity assertion but it is GREEN on
    the base (the content overflowed there too; it simply could not be
    scrolled). The facts that fail on the base are ``allow_vertical_scroll``
    (False), ``scroll_y`` after ``scroll_end`` (0.0), and the last child --
    the pager -- staying below the fold.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)
        assert screen._notes_state.compact is False

        assert lst.max_scroll_y > 0, (
            f"sanity: the tree fits at {size}; the fixture is too small to test"
        )
        assert lst.allow_vertical_scroll is True, (
            f"{size}: #library-notes-list is not a scroll owner "
            f"(overflow_y={lst.styles.overflow_y}, region={lst.region})"
        )

        lst.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        assert lst.scroll_y > 0, f"{size}: scroll_end left scroll_y at {lst.scroll_y}"

        last = lst.children[-1]
        assert lst.region.contains_region(last.region), (
            f"{size}: the list's last row ({last.id}) is still outside the "
            f"list after scroll_end: {last.region} vs {lst.region}"
        )

        # AC#3's wide half, on the same boot: the one-cell scrollbar costs a
        # column, and the full-width rows must still fit beside it.
        _assert_never_scrolls_horizontally(size, screen, lst)


def _assert_never_scrolls_horizontally(size, screen, lst) -> None:
    assert lst.max_scroll_x == 0, (
        f"{size}: horizontal overflow {lst.virtual_size.width} > "
        f"{lst.container_size.width}"
    )
    assert lst.allow_horizontal_scroll is False
    for row in _rows(screen):
        assert row.region.right <= lst.region.right, (
            f"{size}: row {row.id} overruns the list: {row.region} vs {lst.region}"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("size", WIDE_SIZES, ids=lambda s: f"{s[0]}x{s[1]}")
async def test_thirtieth_focused_row_is_visible(size) -> None:
    """AC#2: Down past the fold scrolls the focused row in; Enter opens it.

    Also: Tab from the filter, with the list scrolled to its end, lands on
    the first row -- which is then above the fold -- and reveals it.
    """
    app = _build_test_app()
    _seed_conversations(app, _two_conversations(), notes=_many_notes())
    host = LibraryHarness(app)

    async with host.run_test(size=size) as pilot:
        screen, lst = await _open_notes_tree(host, pilot)

        _rows(screen)[0].focus()
        await pilot.pause()
        for _ in range(DOWN_PRESSES):
            await pilot.press("down")
        await pilot.pause()
        await pilot.pause()

        focused = screen.focused
        assert focused is not None and focused.has_class("library-notes-row")
        assert screen.region.contains_region(focused.region), (
            f"{size}: the focused row {focused.id} is off-screen at "
            f"{focused.region} (screen {screen.region})"
        )
        assert lst.region.contains_region(focused.region), (
            f"{size}: the focused row {focused.id} is outside the list at "
            f"{focused.region} (list {lst.region}, scroll_y={lst.scroll_y})"
        )

        expected_note_id = focused.note_id
        await pilot.press("enter")
        await _wait_for_selector(screen, pilot, "#library-note-title")
        assert screen._notes_state.selected_note_id == expected_note_id

        # Back to the list: scroll it to the end, then Tab from the filter
        # must land on (and reveal) the first row above the fold.
        screen.query_one("#library-row-browse-notes").press()
        screen, lst = await _open_notes_tree(host, pilot)
        lst.scroll_end(animate=False, immediate=True)
        await pilot.pause()
        first = _rows(screen)[0]
        assert not lst.region.contains_region(first.region), (
            "sanity: the first row should be above the fold after scroll_end"
        )
        screen.query_one("#library-notes-filter").focus()
        await pilot.pause()
        for _ in range(12):
            await pilot.press("tab")
            if screen.focused is not None and screen.focused.has_class(
                "library-notes-row"
            ):
                break
        await pilot.pause()
        await pilot.pause()
        focused = screen.focused
        assert focused is not None and focused.has_class("library-notes-row"), (
            f"Tab never reached a note row (focused={getattr(focused, 'id', None)})"
        )
        assert lst.region.contains_region(focused.region), (
            f"{size}: Tab focused {focused.id} but left it outside the list: "
            f"{focused.region} vs {lst.region} (scroll_y={lst.scroll_y})"
        )
