"""task-32306: the Details rail's wrapped rows hang under their first line.

User ruling on the blocked Workspace ▸ Handoff row ("keep the wrap, and indent
the continuation lines, so it's clear what they correspond to").

The mounted pin reads the PAINTED strips rather than the renderable: the indent
only exists once the row's real width is known, and the first attempt at this
fix (a custom Rich renderable) measured as ONE line through Textual 8's visual
protocol -- it painted the hung lines and then clipped them, which a
``.renderable`` probe would have called a pass.
"""

from __future__ import annotations

import pytest
from textual.widgets import Static

from Tests.UI.test_library_shell import (
    LibraryProductionCSSHarness,
    _active_library_screen,
    _build_test_app,
    _wait_for_library_shell,
    _wait_for_selector,
)
from Tests.UI.test_post_release_workspaces_library_depth import _open_library_details
from tldw_chatbook.Widgets.Library.library_rail import (
    LIBRARY_DETAILS_CONTINUATION_PAD,
    library_dim_label_text,
    library_hang_details_row,
)

#: The Details column measured at 235 columns (task-32230's own number,
#: re-measured live for this task).
DETAILS_COLUMN_CELLS = 34


def _painted_rows(app, region) -> list[str]:
    """The plain text the compositor actually painted inside ``region``."""
    strips = app.screen._compositor.render_strips()
    rows: list[str] = []
    for y in range(region.y, min(region.bottom, len(strips))):
        row = "".join(segment.text for segment in strips[y])
        rows.append(row[region.x : region.right])
    return rows


def test_a_blocked_handoff_row_hangs_its_continuation_lines_at_34_cells() -> None:
    """The row this task was filed about, at the width it was measured at.

    Pinned as a function because the blocked state itself is not reachable
    from the mounted harness on this base (the seeded cross-workspace sources
    never register -- ``test_post_release_workspaces_library_depth`` is red for
    that reason before this branch touches anything), so the mounted pin below
    covers the painter and this covers the sentence.
    """
    row = library_dim_label_text(
        "Handoff",
        "24 items can't be used in Console yet · not in this workspace · "
        "Copy or link them into this workspace",
    )
    hung = library_hang_details_row(row, DETAILS_COLUMN_CELLS)
    lines = hung.plain.split("\n")

    assert len(lines) > 1, lines
    assert lines[0].startswith("Handoff · "), lines
    for line in lines[1:]:
        assert line.startswith(LIBRARY_DETAILS_CONTINUATION_PAD), lines
    # Nothing is dropped, truncated or re-ordered by the indent.
    assert " ".join(line.strip() for line in lines) == row.plain
    # Every line still fits the column it is painted in.
    assert max(len(line) for line in lines) <= DETAILS_COLUMN_CELLS, lines


def test_a_row_that_fits_its_width_keeps_the_renderable_it_was_given() -> None:
    """No wrap, no change -- so the rows that never wrapped are untouched."""
    row = library_dim_label_text("Active", "Local Default")
    assert library_hang_details_row(row, DETAILS_COLUMN_CELLS) is row
    assert library_hang_details_row("Status", DETAILS_COLUMN_CELLS) == "Status"


# Only the mounted case is async -- a module-level ``pytestmark`` would warn
# once per sync test in this file (review round 1, F8).
@pytest.mark.asyncio
@pytest.mark.parametrize("size", [(235, 52), (60, 24)], ids=("wide", "narrow"))
async def test_a_wrapped_details_row_paints_its_continuations_indented(size) -> None:
    """Through the product path: mounted, painted, at both supported widths."""
    app = _build_test_app()
    host = LibraryProductionCSSHarness(app)

    async with host.run_test(size=size) as pilot:
        screen = _active_library_screen(host)
        await _wait_for_library_shell(screen, pilot)
        await _open_library_details(screen, pilot)
        await _wait_for_selector(screen, pilot, "#library-workspaces-handoff")

        row = screen.query_one("#library-workspaces-handoff", Static)
        # The Details body runs past the rail's fold; a row that is not on
        # screen paints nothing at all, and an empty capture would pass every
        # assertion below for the wrong reason.
        row.scroll_visible(animate=False)
        await pilot.pause()
        await pilot.pause()
        painted = [line for line in _painted_rows(host, row.region) if line.strip()]
        print(f"MEASURED handoff row at {size}: {row.region!r} -> {painted!r}")

        if size[0] > 100:
            # At 235 columns the rail's Details column is 34 cells and this
            # sentence genuinely wraps. The narrow layout hands the rail the
            # whole screen, so there the same row fits on one line -- which is
            # the other half of the rule, and is asserted by falling through.
            assert len(painted) > 1, (
                "this row no longer wraps at 34 cells, so the pin stopped "
                f"testing anything: {painted!r}"
            )
        first_indent = len(painted[0]) - len(painted[0].lstrip())
        assert painted[0].lstrip().startswith("Handoff · "), painted
        for line in painted[1:]:
            indent = len(line) - len(line.lstrip())
            assert indent == first_indent + len(LIBRARY_DETAILS_CONTINUATION_PAD), (
                f"continuation line is not hung under the first: {line!r} "
                f"in {painted!r}"
            )


@pytest.mark.asyncio
async def test_a_row_re_hangs_in_both_directions_as_its_column_changes() -> None:
    """The resize guard must skip repaints, not updates that change the text.

    ``on_resize`` returns early when the hung form is unchanged (review round
    1, F3). That guard is exactly the kind that can freeze a row at its first
    width, so the widening leg is asserted as well as the narrowing one.
    """
    from textual.app import App, ComposeResult
    from textual.containers import Vertical

    from tldw_chatbook.Widgets.Library.library_rail import (
        LibraryDetailsRow,
        library_dim_label_text,
    )

    source = library_dim_label_text(
        "Handoff", "24 items can't be used in Console yet · not in this workspace"
    )

    class _Host(App):
        CSS = "#box { width: 36; height: auto; } .row { padding: 0 1; }"

        def compose(self) -> ComposeResult:
            with Vertical(id="box"):
                yield LibraryDetailsRow(source, id="row", classes="row")

    app = _Host()
    async with app.run_test(size=(140, 24)) as pilot:
        row = app.query_one("#row", LibraryDetailsRow)
        assert row.region.height > 1, row.region
        assert row.content is source, "the caller's renderable must survive"

        app.query_one("#box").styles.width = 120
        await pilot.pause()
        await pilot.pause()
        assert row.region.height == 1, row.region

        app.query_one("#box").styles.width = 24
        await pilot.pause()
        await pilot.pause()
        assert row.region.height > 1, row.region
        painted = [
            line
            for line in _painted_rows(app, row.region)
            if line.strip()
        ]
        for line in painted[1:]:
            assert line.startswith(f" {LIBRARY_DETAILS_CONTINUATION_PAD}"), painted
