"""Tests for `DetailValueRow` / `DetailGroup` (schedules-redesign PR-1, Task 1).

Rendered (painted) output only, never a stored attribute the widget might
ignore -- last program's lesson. Text content and single-line-ness come from
`Widget.render_line(0)` (the exact Strip Textual paints for that row, after
CSS text-align/text-overflow/wrap is applied); colour comes from the
compositor's own painted segments (`app.screen._compositor.render_strips`),
mirroring the precedent in `Tests/UI/test_model_artifact_widgets.py`.
"""

from __future__ import annotations

import pytest
from rich.text import Text
from textual.app import App, ComposeResult
from textual.widgets._collapsible import CollapsibleTitle

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET, ConsolidatedCSSApp
from tldw_chatbook.Widgets.detail_value_row import DetailGroup, DetailValueRow


def _painted_color(app: App, widget) -> object:
    """Return the foreground colour of the first non-blank glyph painted
    inside `widget`'s own region, read from the compositor's own output."""
    for y in range(widget.region.y, widget.region.bottom):
        cursor = 0
        for segment in app.screen._compositor.render_strips()[y]:
            next_cursor = cursor + segment.cell_length
            overlaps = cursor < widget.region.right and next_cursor > widget.region.x
            if overlaps and segment.text.strip() and segment.style is not None:
                color = segment.style.color
                if color is not None:
                    return color
            cursor = next_cursor
    raise AssertionError(f"no painted glyphs inside {widget.region!r}")


class _RowHarness(ConsolidatedCSSApp):
    """Hosts one or more rows directly, mirroring the standalone-component
    harness pattern in `test_console_inspector_section.py`.

    `CSS_PATH` adds the app bundle (not just the screen sheets
    `ConsolidatedCSSApp` loads by default) because `DetailValueRow`'s own
    `BUNDLED_CSS` references `$ds-*` design tokens, which are only defined
    inside `tldw_cli_modular.tcss` (`css/core/_variables.tcss`, concatenated
    in first) -- Textual pools `$variable` definitions across every source
    feeding one `Stylesheet`, so the token source and its usage must share
    a harness. `ConsolidatedCSSApp.__init__` re-merges the screen sheets
    around whatever `CSS_PATH` a subclass supplies.
    """

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self, *rows: DetailValueRow) -> None:
        super().__init__()
        self._rows = rows

    def compose(self) -> ComposeResult:
        yield from self._rows


class _GroupHarness(ConsolidatedCSSApp):
    """See `_RowHarness` -- `DetailGroup`'s `BUNDLED_CSS` also uses `$ds-*`."""

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self, group: DetailGroup) -> None:
        super().__init__()
        self._group = group

    def compose(self) -> ComposeResult:
        yield self._group


@pytest.mark.asyncio
async def test_label_and_value_are_painted_with_value_right_aligned():
    row = DetailValueRow("Status", "Waiting", id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)):
        label = row.query_one(".detail-value-row-label")
        value = row.query_one(".detail-value-row-value")
        assert label.render_line(0).text.strip() == "Status"
        painted_value = value.render_line(0).text
        # Right-aligned: the value's own glyphs sit flush against the
        # right edge of its box, not the left (padded with spaces on the
        # left instead of the right).
        assert painted_value.rstrip() == painted_value
        assert painted_value.strip() == "Waiting"


@pytest.mark.asyncio
async def test_affordance_glyph_is_painted_and_dimmer_than_the_value():
    with_affordance = DetailValueRow("Repeat", "Weekly", affordance=True, id="a")
    without_affordance = DetailValueRow("Repeat", "Weekly", id="b")
    app = _RowHarness(with_affordance, without_affordance)
    async with app.run_test(size=(40, 6)):
        affordance = with_affordance.query_one(".detail-value-row-affordance")
        value = with_affordance.query_one(".detail-value-row-value")
        assert affordance.render_line(0).text.strip() == "▾"
        assert _painted_color(app, affordance) != _painted_color(app, value), (
            "affordance glyph must be visually dimmer than the value it sits "
            "beside, not the same colour"
        )
        assert len(without_affordance.query(".detail-value-row-affordance")) == 0


@pytest.mark.asyncio
async def test_update_value_refreshes_the_same_widget_in_place():
    row = DetailValueRow("Status", "Waiting", id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        value = row.query_one(".detail-value-row-value")
        row.update_value("Running")
        await pilot.pause()
        # Same mounted widget instance -- no recompose -- now painting the
        # new text.
        assert row.query_one(".detail-value-row-value") is value
        assert value.render_line(0).text.strip() == "Running"


@pytest.mark.asyncio
async def test_error_slot_hidden_by_default_then_shown_and_cleared():
    row = DetailValueRow("Schedule", "Every Monday", value_id="row-schedule")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 6)) as pilot:
        error = row.query_one("#row-schedule-error")
        assert error.region.height == 0, "error slot must start painted-hidden"

        row.show_error("Invalid cron expression")
        await pilot.pause()
        error = row.query_one("#row-schedule-error")
        assert error.region.height > 0
        assert "Invalid cron expression" in error.render_line(0).text

        row.clear_error()
        await pilot.pause()
        error = row.query_one("#row-schedule-error")
        assert error.region.height == 0, "clear_error must re-hide the slot"


@pytest.mark.asyncio
async def test_str_value_renders_literally_not_as_rich_markup():
    row = DetailValueRow("Title", "[bold]Weekly sync[/bold]", id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(50, 5)):
        value = row.query_one(".detail-value-row-value")
        painted = value.render_line(0).text.strip()
        assert painted == "[bold]Weekly sync[/bold]", (
            "a plain str value must never be interpreted as Rich markup -- "
            f"got {painted!r}"
        )


@pytest.mark.asyncio
async def test_text_value_is_accepted_and_rendered():
    row = DetailValueRow("Owner", Text("server", style="bold"), id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)):
        value = row.query_one(".detail-value-row-value")
        assert value.render_line(0).text.strip() == "server"


@pytest.mark.asyncio
async def test_long_value_ellipsizes_and_never_wraps_the_row():
    long_value = "A" * 200
    row = DetailValueRow("Title", long_value, id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(30, 5)):
        assert row.region.height == 1, "an overlong value must not wrap the row"
        value = row.query_one(".detail-value-row-value")
        painted = value.render_line(0).text
        assert painted.strip().endswith("…"), (
            f"expected an ellipsis marker, got {painted!r}"
        )
        assert long_value not in painted


@pytest.mark.asyncio
async def test_group_collapse_toggle_hides_and_reveals_the_row():
    child = DetailValueRow("Status", "Waiting", id="child-row")
    group = DetailGroup(child, title="Reminder", id="group")
    app = _GroupHarness(group)
    async with app.run_test(size=(40, 10)) as pilot:
        assert group.collapsed is False
        assert child.region.height > 0, "group starts open (collapsed=False)"

        await pilot.click(CollapsibleTitle)
        await pilot.pause()
        assert group.collapsed is True
        assert child.region.height == 0, "collapsing must hide the body row"

        await pilot.click(CollapsibleTitle)
        await pilot.pause()
        assert group.collapsed is False
        assert child.region.height > 0, "expanding again must repaint the row"


@pytest.mark.asyncio
async def test_group_starts_collapsed_when_requested():
    child = DetailValueRow("Status", "Waiting", id="child-row")
    group = DetailGroup(child, title="Reminder", collapsed=True, id="group")
    app = _GroupHarness(group)
    async with app.run_test(size=(40, 10)):
        assert group.collapsed is True
        assert child.region.height == 0
