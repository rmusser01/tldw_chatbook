"""The media canvas's multi-row action grammar (task-30043).

Critique 2026-09-03 P1: the items pane sits at ~40-44 cols in every real
shell layout, so the old single six-button row chopped to ``t so E Tr R Se``
and select mode's bulk actions rendered as bare ``○ ○ ○``. The multi-row
grammar is THE grammar — every row's label budget (including the "○ "
disabled forms) fits the pane's 40-col floor, so labels are always words.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button, Static

from tldw_chatbook.Library.library_media_state import (
    LibraryMediaCanvasState,
    LibraryMediaRow,
)
from tldw_chatbook.Widgets.Library.library_media_canvas import LibraryMediaCanvas
from Tests.UI.consolidated_css import ConsolidatedCSSApp


def _browse_state() -> LibraryMediaCanvasState:
    rows = (
        LibraryMediaRow(
            media_id="1",
            title="First item",
            media_type="document",
            secondary="document · today",
        ),
    )
    return LibraryMediaCanvasState(
        rows=rows,
        type_options=(None, "document"),
        active_type=None,
        status_copy="",
        empty_copy="No media yet.",
        selected_id="",
        preview_lines=(),
        count=1,
    )


def _select_state(
    *, selected_count: int = 0, confirming: bool = False
) -> LibraryMediaCanvasState:
    rows = (
        LibraryMediaRow(
            media_id="1",
            title="First item",
            media_type="document",
            secondary="document · today",
            checked=selected_count > 0,
        ),
        LibraryMediaRow(
            media_id="2",
            title="Second item",
            media_type="document",
            secondary="document · today",
            checked=False,
        ),
    )
    return LibraryMediaCanvasState(
        rows=rows,
        type_options=(None, "document"),
        active_type=None,
        status_copy="",
        empty_copy="No media yet.",
        selected_id="",
        preview_lines=(),
        count=2,
        select_mode=True,
        selected_count=selected_count,
        confirming_bulk_delete=confirming,
    )


class _CanvasApp(ConsolidatedCSSApp):
    def __init__(self, state: LibraryMediaCanvasState) -> None:
        super().__init__()
        self._state = state

    def compose(self):
        yield LibraryMediaCanvas(canvas=self._state, id="library-media-canvas")


@pytest.mark.asyncio
async def test_browse_actions_split_into_labeled_rows():
    """Choosers / plain actions / Review these each get a fitting row."""
    app = _CanvasApp(_browse_state())
    async with app.run_test(size=(50, 34)) as pilot:
        await pilot.pause()
        assert app.query("#library-media-toolbar-choosers")
        assert app.query("#library-media-toolbar-actions")
        assert app.query("#library-media-toolbar-review")
        # Full labels survive: no chopped fragments.
        assert str(app.query_one("#library-media-review", Button).label) == (
            "Review these"
        )
        assert str(app.query_one("#library-media-export", Button).label) == (
            "Export…"
        )
        assert str(
            app.query_one("#library-media-trash-open", Button).label
        ) == "Trash"
        assert str(
            app.query_one("#library-media-select-toggle", Button).label
        ) == "Select"


@pytest.mark.asyncio
async def test_browse_action_rows_fit_the_pane_floor():
    """At 40 cols every action stays inside the canvas — no clipped buttons."""
    app = _CanvasApp(_browse_state())
    async with app.run_test(size=(40, 34)) as pilot:
        await pilot.pause()
        canvas = app.query_one("#library-media-canvas", LibraryMediaCanvas)
        right = canvas.region.x + canvas.region.width
        for selector in (
            "#library-media-type-filter",
            "#library-media-sort",
            "#library-media-export",
            "#library-media-trash-open",
            "#library-media-review",
            "#library-media-select-toggle",
        ):
            button = app.query_one(selector, Button)
            assert button.region.width > 0, selector
            assert button.region.x + button.region.width <= right, selector


@pytest.mark.asyncio
async def test_select_mode_bulk_actions_are_words_not_markers():
    """Disabled bulk actions read '○ Export', never a bare '○'."""
    app = _CanvasApp(_select_state(selected_count=0))
    async with app.run_test(size=(50, 34)) as pilot:
        await pilot.pause()
        export = app.query_one("#library-media-export-selected", Button)
        review = app.query_one("#library-media-review-selected", Button)
        delete = app.query_one("#library-media-delete-selected", Button)
        assert str(export.label) == "○ Export"
        assert str(review.label) == "○ Review"
        assert str(delete.label) == "○ Delete"
        # The in-place count-crossing patch rebuilds from the same short base.
        assert export._library_disabled_marker_base == "Export"
        # The full meaning still rides on the F-018 tooltip.
        assert "Select one or more items" in str(export.tooltip)
        # Summary keeps the full Select-all sentence.
        assert str(app.query_one("#library-media-select-all", Button).label) == (
            "Select all 2 shown"
        )
        # Delete is isolated on its own danger row, never adjacent to another
        # action (task-2853's rule, upgraded).
        danger_row = app.query_one("#library-media-select-danger")
        assert len(danger_row.query(Button)) == 1
        actions_row = app.query_one("#library-media-select-actions")
        assert len(actions_row.query(Button)) == 3  # Clear, Export, Review


@pytest.mark.asyncio
async def test_select_mode_rows_fit_the_pane_floor():
    app = _CanvasApp(_select_state(selected_count=1))
    async with app.run_test(size=(40, 34)) as pilot:
        await pilot.pause()
        canvas = app.query_one("#library-media-canvas", LibraryMediaCanvas)
        right = canvas.region.x + canvas.region.width
        for selector in (
            "#library-media-select-all",
            "#library-media-select-clear",
            "#library-media-export-selected",
            "#library-media-review-selected",
            "#library-media-delete-selected",
        ):
            button = app.query_one(selector, Button)
            assert button.region.width > 0, selector
            assert button.region.x + button.region.width <= right, selector


@pytest.mark.asyncio
async def test_confirm_copy_wraps_inside_the_pane():
    app = _CanvasApp(_select_state(selected_count=2, confirming=True))
    async with app.run_test(size=(50, 34)) as pilot:
        await pilot.pause()
        copy = app.query_one("#library-media-bulk-delete-confirm-copy", Static)
        canvas = app.query_one("#library-media-canvas", LibraryMediaCanvas)
        assert copy.region.width <= canvas.region.width
        assert copy.region.height >= 2  # the safety sentence wrapped, not clipped


@pytest.mark.asyncio
async def test_long_type_values_are_capped_in_the_chooser_label():
    """A long stored media type must not re-overflow the chooser row.

    Qodo #2350: type values are data; "type: presentation" + "sort: Title
    A-Z" exceeded the pane's ~34 usable cells. The LABEL caps the value
    (full value stays in the tooltip and in the chooser strip itself).
    """
    state = _browse_state()
    import dataclasses

    state = dataclasses.replace(
        state,
        type_options=(None, "presentation-deck"),
        active_type="presentation-deck",
    )
    app = _CanvasApp(state)
    async with app.run_test(size=(40, 34)) as pilot:
        await pilot.pause()
        type_button = app.query_one("#library-media-type-filter", Button)
        label = str(type_button.label)
        assert "presentation-deck" not in label  # capped in the label...
        assert label.endswith("…")
        assert "presentation-deck" in str(type_button.tooltip)  # ...not the tooltip
        canvas = app.query_one("#library-media-canvas", LibraryMediaCanvas)
        right = canvas.region.x + canvas.region.width
        sort_button = app.query_one("#library-media-sort", Button)
        for button in (type_button, sort_button):
            assert button.region.x + button.region.width <= right
