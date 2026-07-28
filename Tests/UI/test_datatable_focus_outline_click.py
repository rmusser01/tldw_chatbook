"""The global focus outline must not eat a focused `DataTable`'s clicks — TASK-1160.

`core/_reset.tcss` carries the app-wide keyboard-focus fallback::

    *:focus { outline: solid $ds-focus-accent; }

Textual paints an ``outline`` OVER the widget's outermost rendered lines
rather than around them -- that is precisely what distinguishes it from
``border``, which costs geometry. The segments it overwrites lose the
``{"row", "column"}`` Rich style metadata, and ``DataTable._on_click``
resolves the clicked cell from nothing else::

    meta = event.style.meta
    if "row" not in meta or "column" not in meta:
        return

so a click on the bottom row of a *focused* table is silently dropped. It
reads as intermittent rather than broken because ``MouseDown`` focuses the
table before the ``Click`` is resolved -- the first click on a blurred table
still lands, and the second one onto the bottom row is the one that does
nothing.

Measured in a bare Textual app (no tldw CSS at all, the outline rule the only
variable), 6-row table, region height 7::

    no outline            outline
    click y=5 -> row 4    click y=5 -> row 4
    click y=6 -> row 5    click y=6 -> row 0   <- last row dead

TASK-1105 fixed this for the Watchlists screen and TASK-1034 fixed the header
half of it for the Evals results grid, both with screen-local ``outline:
none`` rules. TASK-1160 promoted the fix to a bare ``DataTable:focus`` type
selector in ``components/_lists.tcss`` so no call site has to opt in, and
removed both local copies.

These tests deliberately use a table with **no** id or class of its own,
under the **production** stylesheet. A bare ``App`` with no CSS cannot
reproduce the defect at all -- the outline is the cause -- so the bundle is
loaded, and ``test_the_global_focus_outline_fallback_is_still_in_the_bundle``
guards against this suite quietly going vacuous if that fallback is ever
dropped.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.geometry import Offset
from textual.widgets import DataTable

import tldw_chatbook

_BUNDLED_CSS_PATH = Path(tldw_chatbook.__file__).parent / "css" / "tldw_cli_modular.tcss"

ROW_COUNT = 6
LAST_ROW = ROW_COUNT - 1


class _ProductionCssTableHarness(App):
    """A plain `DataTable` under the real app stylesheet.

    No id, no classes: nothing here can be matched by a screen-scoped rule,
    so whatever these tests observe is what the ~40 `DataTable` call sites
    across the app get by default.
    """

    CSS_PATH = str(_BUNDLED_CSS_PATH)

    def compose(self) -> ComposeResult:
        yield DataTable()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.add_column("Name", key="name")
        table.add_column("Kind", key="kind")
        for index in range(ROW_COUNT):
            table.add_row(f"row-{index}", f"kind-{index}", key=str(index))


def test_the_global_focus_outline_fallback_is_still_in_the_bundle() -> None:
    """Guards the rest of this file against becoming vacuous.

    Every assertion below is only meaningful while `*:focus { outline: solid
    ... }` is actually in force. If that fallback is ever removed wholesale,
    the click tests would pass for the wrong reason -- this one fails loudly
    instead and says so.
    """
    bundle = _BUNDLED_CSS_PATH.read_text(encoding="utf-8")
    assert "outline: solid $ds-focus-accent;" in bundle, (
        "core/_reset.tcss's app-wide focus-outline fallback is gone from the "
        "bundle -- the TASK-1160 tests in this file no longer prove anything. "
        "Either restore it or retire this file."
    )


@pytest.mark.asyncio
async def test_clicking_the_last_row_of_a_focused_table_moves_the_cursor() -> None:
    """AC#1. The regression: bottom row of an ALREADY-focused table.

    The cursor is parked on row 0 first, so only the click itself can move
    it -- and the table is focused before the click, which is the state a
    user is in from their second click onwards.
    """
    app = _ProductionCssTableHarness()
    async with app.run_test(size=(60, 14)) as pilot:
        table = app.query_one(DataTable)
        table.focus()
        await pilot.pause()
        assert table.has_focus

        table.move_cursor(row=0, column=0)
        await pilot.pause()
        assert table.cursor_coordinate.row == 0

        # `outline` costs no geometry, so the region is the same height with
        # or without it -- the bottom line is the last row either way.
        bottom_line = table.region.height - 1
        await pilot.click(table, offset=Offset(2, bottom_line))
        await pilot.pause()

        assert table.cursor_coordinate.row == LAST_ROW, (
            f"clicking the bottom line (y={bottom_line}) of a focused table "
            f"left the cursor on row {table.cursor_coordinate.row}; the global "
            "focus outline is overwriting that line's row/column metadata "
            "(see components/_lists.tcss, TASK-1160)"
        )


@pytest.mark.asyncio
async def test_every_row_of_a_focused_table_is_clickable() -> None:
    """The perimeter, not just the last row: the outline also kills column
    x=0 on every line and overwrites the header line outright."""
    app = _ProductionCssTableHarness()
    async with app.run_test(size=(60, 14)) as pilot:
        table = app.query_one(DataTable)
        table.focus()
        await pilot.pause()

        landed: list[int] = []
        for row in range(ROW_COUNT):
            table.move_cursor(row=0, column=1)
            await pilot.pause()
            # +1 for the header line that occupies y=0.
            await pilot.click(table, offset=Offset(0, row + 1))
            await pilot.pause()
            landed.append(table.cursor_coordinate.row)

        assert landed == list(range(ROW_COUNT)), (
            f"clicking column 0 of each row of a focused table landed on "
            f"{landed}, expected {list(range(ROW_COUNT))}"
        )


@pytest.mark.asyncio
async def test_a_focused_table_still_shows_a_visible_focus_affordance() -> None:
    """AC#2. Suppressing the outline must not leave keyboard users blind.

    Before TASK-1160 `components/_lists.tcss` restyled `.datatable--cursor`
    unconditionally, which flattened Textual's own focused/blurred cursor
    distinction -- so the outline really was the only cue, and simply
    deleting it would have traded one defect for another. Two cues replace
    it, because the cursor can be switched off (`cursor_type="none"`) or
    scrolled out of view while the header is always on screen.
    """
    app = _ProductionCssTableHarness()
    async with app.run_test(size=(60, 14)) as pilot:
        table = app.query_one(DataTable)

        table.blur()
        await pilot.pause()
        blurred_cursor = table.get_component_styles("datatable--cursor").background
        blurred_header = table.get_component_styles("datatable--header").background

        table.focus()
        await pilot.pause()
        focused_cursor = table.get_component_styles("datatable--cursor").background
        focused_header = table.get_component_styles("datatable--header").background

        assert focused_cursor != blurred_cursor, (
            "a focused table's cursor cell looks identical to a blurred "
            f"one's ({focused_cursor}) -- no focus affordance"
        )
        assert focused_header != blurred_header, (
            "a focused table's column header looks identical to a blurred "
            f"one's ({focused_header}) -- no focus affordance"
        )


@pytest.mark.asyncio
async def test_a_focused_table_still_renders_its_column_header() -> None:
    """TASK-1034, generalised: the outline's top edge replaced the header
    line of any focused table. Rendered here through the real compositor,
    not read off the model."""
    app = _ProductionCssTableHarness()
    async with app.run_test(size=(60, 14)) as pilot:
        table = app.query_one(DataTable)
        table.focus()
        await pilot.pause()

        header_line = app.screen._compositor.render_strips()[table.region.y]
        header_text = header_line.text

        assert "Name" in header_text and "Kind" in header_text, (
            f"the focused table's header line rendered as {header_text!r}; "
            "the focus outline has painted over it"
        )
