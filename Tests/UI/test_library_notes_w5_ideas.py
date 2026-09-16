"""Library ▸ Notes critique #4, wave 5 -- the three accepted IDEAS (G8).

Tasks 32640, 32641 and 32642: the ideas task-32627 accepted. See
``backlog/tasks/task-<id>*.md`` for the acceptance criteria; each test names
the task it pins.

Everything here reads production output -- the real mounted screen at the
critique's own terminal sizes, the real session draft, the real import
controller -- never a value the test just handed the code. Where a walk
matters it is walked with ``pilot.press("tab")``, never ``widget.focus()``.
"""

from __future__ import annotations

import re

import pytest
from textual.widgets import Button, Input, Static

from Tests.UI.test_library_shell import (
    _active_library_screen,
    _wait_for_condition,
    _wait_for_library_shell,
)
from Tests.UI.test_library_notes_w4_editor import (
    _build_notes_host,
    _open_first_note,
)
from tldw_chatbook.Widgets.Library.library_notes_canvas import (
    library_note_property_block,
)

pytestmark = pytest.mark.asyncio

#: The critique's own terminal sizes (task-32614 measured at these three).
WIDE = (235, 52)
COMPACT = (100, 30)
NARROW = (60, 20)

#: A note whose Created/Modified/Version/Words spell out four properties --
#: the timestamps are what the joined line could not fit in a compact pane.
_DATED_NOTE = [
    {
        "id": "note-dated",
        "title": "Garden redesign",
        "content": "Body text with several words in it\n",
        "version": 3,
        "created_at": "2026-07-01T03:00:00+00:00",
        "last_modified": "2026-09-14T11:02:00+00:00",
    }
]


def _meta_rows(screen) -> list[str]:
    """Info's Properties Static, row by row, as it is rendered."""
    meta = screen.query_one("#library-note-context-meta", Static)
    return str(meta.renderable).split("\n")


async def _open_info(screen, pilot) -> None:
    screen.query_one("#library-note-context", Button).press()
    await _wait_for_condition(
        pilot,
        lambda: screen.query_one("#library-note-context-region").display,
        message="Info never opened.",
    )
    await pilot.pause()


# --- task-32642 AC#1/#4: one row per property, nothing dropped -------------


async def test_info_gives_every_property_its_own_row_at_235x52():
    """task-32642 AC#1/AC#4.

    Born red against the unfixed tree, where the whole Static was ONE row
    155 columns wide inside a 36-row pane:
    ``assert ['Created 2026-06-30 20:00 · 10w ago · Modified 2026-09-14
    04:02 · 1d ago · v3 · 7 words'] == ['Created', 'Modified', 'Version',
    'Words']``. Four facts shared a row; the pane had 21 spare ones.
    """
    host = _build_notes_host(notes=_DATED_NOTE)
    async with host.run_test(size=WIDE) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        rows = _meta_rows(screen)
        labels = [row.split("  ")[0].strip() for row in rows]
        assert labels == ["Created", "Modified", "Version", "Words"], rows
        # AC#4: arrangement, not loss -- every fact the joined line carried
        # is still on screen, and the Static is as tall as it has rows.
        meta = screen.query_one("#library-note-context-meta", Static)
        assert meta.region.height == len(rows), (rows, meta.region.height)
        # Local-zone rendered, so the DATE is not asserted literally -- the
        # shape is: an absolute stamp beside a relative age, per row.
        assert all(
            re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} · \S+ ago$", row)
            for row in rows[:2]
        ), rows
        assert rows[2].endswith("v3"), rows
        assert rows[3].endswith("words"), rows


async def test_the_compact_property_block_keeps_one_column_and_no_truncation():
    """task-32642 AC#2.

    The pane is 46 columns wide at 100x30 and the compact sheet pinned this
    Static to ``height: 1``, so the 84-character joined line was cut. One
    property per row, unpadded, and every value present in full.

    Born red on the unfixed tree -- and the RED is worse than the finding
    said. Measured on 67bfde41d1 with the Python AND the three stylesheets
    reverted, ``#library-note-context-meta`` painted exactly one row::

        Created 2026-06-30 20:00 · 10w ago · Modified

    The Modified value, the version and the word count were all off the
    pane at 100x30: three of Info's four properties were unreadable there,
    not merely crowded.
    """
    host = _build_notes_host(notes=_DATED_NOTE)
    async with host.run_test(size=COMPACT) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)
        await _open_info(screen, pilot)

        rows = _meta_rows(screen)
        assert [row.split(" ", 1)[0] for row in rows] == [
            "Created",
            "Modified",
            "Version",
            "Words",
        ], rows
        # No alignment padding in the compact spelling: one column.
        assert all("  " not in row for row in rows), rows
        # Values arrive whole: the age still ends each timestamp row, which
        # the truncated single line could not do.
        assert all(row.endswith(" ago") for row in rows[:2]), rows
        assert all(
            re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} · \S+ ago$", row)
            for row in rows[:2]
        ), rows


def test_the_property_block_falls_back_to_the_joined_line_without_pairs():
    """A caller that supplies no pairs still renders the sentence it had.

    This is the vacuity guard for the two tests above: if
    ``library_note_property_block`` returned the joined line for every
    input, they would both fail -- and if it ignored ``compact``, this
    would still pass. Each shape is asserted against an input that can
    only produce it.
    """
    pairs = (("Created", "2026-07-01 03:00 · 10w ago"), ("Words", "6 words"))
    assert library_note_property_block((), "Updated today", compact=False) == (
        "Updated today"
    )
    assert library_note_property_block(pairs, "ignored", compact=True) == (
        "Created 2026-07-01 03:00 · 10w ago\nWords 6 words"
    )
    assert library_note_property_block(pairs, "ignored", compact=False) == (
        "Created  2026-07-01 03:00 · 10w ago\nWords    6 words"
    )


# --- task-32642 AC#3: keywords without opening Info ------------------------


@pytest.mark.parametrize("size", (WIDE, COMPACT, NARROW))
async def test_keywords_are_reachable_and_editable_from_the_editor(size):
    """task-32642 AC#3, at all three critique sizes.

    Born red on the unfixed tree, where ``#library-note-keywords`` was
    mounted inside ``#library-note-wide-utilities`` -- a container
    ``apply_session_state`` sets ``display = False`` unconditionally -- so
    forward Tab from the title went straight to the body and the walk never
    reached it: ``assert 'library-note-keywords' in ['library-note-body',
    ...]``.

    Walked with ``pilot.press("tab")``: a focus-order pin that sets focus
    directly cannot see a control that is displayed but skipped.
    """
    host = _build_notes_host(notes=_DATED_NOTE)
    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_first_note(screen, pilot)

        title = screen.query_one("#library-note-title", Input)
        title.focus()
        await pilot.pause()

        visited: list[str] = []
        for _ in range(4):
            await pilot.press("tab")
            await pilot.pause()
            focused = screen.focused
            visited.append("" if focused is None else (focused.id or ""))
            if visited[-1] == "library-note-keywords":
                break
        assert "library-note-keywords" in visited, visited

        # ...and it EDITS: the canonical draft, not a detached widget value.
        before = screen._library_note_session.snapshot.keywords_text
        await pilot.press("g", "a", "r", "d", "e", "n")
        await pilot.pause()
        after = screen._library_note_session.snapshot.keywords_text
        assert after == f"{before}garden", (before, after)
        # Info was never opened to get here.
        assert screen.query_one("#library-note-context-region").display is False
