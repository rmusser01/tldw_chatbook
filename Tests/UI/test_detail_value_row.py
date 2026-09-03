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
from textual.css.query import NoMatches
from textual.widgets import Input, Select
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


def _relative_luminance(color) -> float:
    """WCAG relative luminance of a compositor-painted colour (0=black,
    1=white) -- same formula as `Tests/UI/test_model_artifact_widgets.py`'s
    contrast helper. Lower means visually dimmer."""
    triplet = color.get_truecolor()

    def channel(value: int) -> float:
        srgb = value / 255
        return srgb / 12.92 if srgb <= 0.04045 else ((srgb + 0.055) / 1.055) ** 2.4

    return (
        0.2126 * channel(triplet.red)
        + 0.7152 * channel(triplet.green)
        + 0.0722 * channel(triplet.blue)
    )


class _RowHarness(ConsolidatedCSSApp):
    """Hosts one or more rows directly, mirroring the standalone-component
    harness pattern in `test_console_inspector_section.py`.

    `CSS_PATH` adds the app bundle (not just the screen sheets
    `ConsolidatedCSSApp` loads by default) because `DetailValueRow`'s
    styling lives in `css/features/_scheduling.tcss`, which is only part of
    the app bundle (`tldw_cli_modular.tcss`, via `build_css.py`'s
    `CSS_MODULES`) -- not the two screen sheets `ConsolidatedCSSApp` loads
    on its own. `ConsolidatedCSSApp.__init__` re-merges the screen sheets
    around whatever `CSS_PATH` a subclass supplies.
    """

    CSS_PATH = str(BUNDLED_STYLESHEET)

    def __init__(self, *rows: DetailValueRow) -> None:
        super().__init__()
        self._rows = rows
        #: `DetailValueRow.Activated` rows received, in post order (PR-3
        #: Task 1) -- Textual routes a message subclass to the handler
        #: named after its dotted qualname (`DetailValueRow.Activated` ->
        #: `on_detail_value_row_activated`), same convention as
        #: `Button.Pressed` -> `on_button_pressed`.
        self.activations: list[DetailValueRow] = []

    def compose(self) -> ComposeResult:
        yield from self._rows

    def on_detail_value_row_activated(self, message: DetailValueRow.Activated) -> None:
        self.activations.append(message.row)


class _GroupHarness(ConsolidatedCSSApp):
    """See `_RowHarness` -- `DetailGroup`'s styling lives in the same
    `_scheduling.tcss` block."""

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
        affordance_luminance = _relative_luminance(_painted_color(app, affordance))
        value_luminance = _relative_luminance(_painted_color(app, value))
        assert affordance_luminance < value_luminance, (
            "affordance glyph must be visually dimmer (lower luminance) than "
            f"the value it sits beside: affordance={affordance_luminance:.3f} "
            f"value={value_luminance:.3f}"
        )
        # The glyph is always MOUNTED (final review F13.1 made
        # `affordance` a settable property, so PR-3 can flip a row
        # without a remount) -- what a row without one must not do is
        # PAINT it.
        hidden = without_affordance.query_one(".detail-value-row-affordance")
        assert hidden.region.width == 0
        assert hidden.display is False


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


@pytest.mark.asyncio
async def test_affordance_can_be_flipped_after_mount_without_a_remount():
    """Final review F13.1: PR-3 flips a row between read-only and editable
    per state, and both consumers hold hard refs to rows assigned inside
    their own `compose()` -- rebuilding the row would mean a remount."""
    row = DetailValueRow("Repeat", "Weekly", id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 6)) as pilot:
        value = row.query_one(".detail-value-row-value")
        glyph = row.query_one(".detail-value-row-affordance")
        assert row.affordance is False
        assert glyph.region.width == 0

        row.affordance = True
        await pilot.pause()
        assert glyph.region.width > 0
        assert glyph.render_line(0).text.strip() == "▾"
        # Same mounted widgets throughout -- no recompose, so the painted
        # value survives the flip.
        assert row.query_one(".detail-value-row-value") is value
        assert value.render_line(0).text.strip() == "Weekly"

        row.affordance = False
        await pilot.pause()
        assert row.affordance is False
        assert glyph.region.width == 0


@pytest.mark.asyncio
async def test_row_carries_its_own_identity_and_focusability():
    """Final review F13.2/F13.3: PR-3 addresses the ROW (open its editor,
    route its error) and spec §12 wants Up/Down row traversal -- neither
    should need `static.parent.parent` or a subclass."""
    plain = DetailValueRow("Repeat", "Weekly", id="plain")
    keyed = DetailValueRow(
        "Repeat", "Weekly", row_key="schedule.cron", can_focus=True, id="keyed"
    )
    app = _RowHarness(plain, keyed)
    async with app.run_test(size=(40, 8)):
        assert plain.row_key is None
        assert plain.can_focus is False, "PR-1 rows stay read-only by default"
        assert keyed.row_key == "schedule.cron"
        assert keyed.can_focus is True
        keyed.focus()
        assert app.focused is keyed


# ---------------------------------------------------------------------------
# schedules-redesign PR-3, Task 1: activation + edit-swap API
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_click_posts_activated_when_affordance_true_and_not_editing():
    row = DetailValueRow("Repeat", "Weekly", affordance=True, id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        await pilot.click(row)
        await pilot.pause()
        assert app.activations == [row]


@pytest.mark.asyncio
async def test_enter_posts_activated_when_row_focused():
    row = DetailValueRow(
        "Repeat", "Weekly", affordance=True, can_focus=True, id="row"
    )
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        row.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.activations == [row]


@pytest.mark.asyncio
async def test_click_on_a_dormant_row_never_posts_activated():
    """PR-1/PR-2 preservation: `affordance` left at its default `False`
    (every current TaskDetail/DefinitionDetail row) must stay a complete
    no-op, exactly as before this row had a click/key handler at all."""
    row = DetailValueRow("Status", "Waiting", id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        await pilot.click(row)
        await pilot.pause()
        assert app.activations == []


@pytest.mark.asyncio
async def test_enter_on_a_focusable_but_non_editable_row_never_posts_activated():
    """`can_focus` and `affordance` are independent flags -- a row can be
    keyboard-focusable (spec §12 traversal) without being editable; Enter
    on it must not activate."""
    row = DetailValueRow(
        "Status", "Waiting", can_focus=True, id="row"
    )
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        row.focus()
        await pilot.pause()
        await pilot.press("enter")
        await pilot.pause()
        assert app.activations == []


@pytest.mark.asyncio
async def test_click_while_an_editor_is_already_open_does_not_reactivate():
    row = DetailValueRow("Repeat", "Weekly", affordance=True, id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        row.begin_edit(Input(id="editor"))
        await pilot.pause()

        await pilot.click(row)
        await pilot.pause()
        assert app.activations == []


@pytest.mark.asyncio
async def test_begin_edit_hides_the_value_and_mounts_a_focused_editor():
    row = DetailValueRow("Repeat", "Weekly", affordance=True, id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        value = row.query_one(".detail-value-row-value")
        assert value.display is True

        editor = Input(id="editor")
        row.begin_edit(editor)
        await pilot.pause()

        assert value.display is False
        assert row.query_one("#editor") is editor
        assert app.focused is editor


@pytest.mark.asyncio
async def test_end_edit_restores_the_value_and_refocuses_the_row_by_default():
    row = DetailValueRow(
        "Repeat", "Weekly", affordance=True, can_focus=True, id="row"
    )
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        value = row.query_one(".detail-value-row-value")
        row.begin_edit(Input(id="editor"))
        await pilot.pause()

        row.end_edit()
        await pilot.pause()

        with pytest.raises(NoMatches):
            row.query_one("#editor")
        assert value.display is True
        assert value.render_line(0).text.strip() == "Weekly"
        assert app.focused is row


@pytest.mark.asyncio
async def test_end_edit_with_restore_focus_false_leaves_focus_alone():
    row = DetailValueRow(
        "Repeat", "Weekly", affordance=True, can_focus=True, id="row"
    )
    other = DetailValueRow("At", "9:00 AM", can_focus=True, id="other")
    app = _RowHarness(row, other)
    async with app.run_test(size=(40, 6)) as pilot:
        row.begin_edit(Input(id="editor"))
        await pilot.pause()
        other.focus()
        await pilot.pause()

        row.end_edit(restore_focus=False)
        await pilot.pause()

        assert app.focused is other


@pytest.mark.asyncio
async def test_begin_edit_is_a_guarded_noop_while_already_editing():
    """One editor at a time -- a second `begin_edit` while one is open must
    not mount its editor at all."""
    row = DetailValueRow("Repeat", "Weekly", affordance=True, id="row")
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        first = Input(id="first-editor")
        second = Input(id="second-editor")
        row.begin_edit(first)
        await pilot.pause()

        row.begin_edit(second)
        await pilot.pause()

        assert row.query_one("#first-editor") is first
        assert second.parent is None, "second editor must never be mounted"
        assert app.focused is first


@pytest.mark.asyncio
async def test_error_line_coexists_with_an_open_editor():
    row = DetailValueRow(
        "Repeat", "Weekly", affordance=True, value_id="row-repeat", id="row"
    )
    app = _RowHarness(row)
    async with app.run_test(size=(40, 6)) as pilot:
        row.begin_edit(Input(id="editor"))
        await pilot.pause()

        row.show_error("Invalid cron expression")
        await pilot.pause()

        error = row.query_one("#row-repeat-error")
        assert error.region.height > 0
        assert "Invalid cron expression" in error.render_line(0).text
        # The editor must still be there -- showing an error must not
        # close it out from under the user.
        assert row.query_one("#editor").display is True


@pytest.mark.asyncio
async def test_affordance_undims_to_the_value_color_when_the_row_has_focus():
    """PR-3 Task 1's live-state CSS: PR-1's resting affordance is dimmer
    than the value (pinned by
    `test_affordance_glyph_is_painted_and_dimmer_than_the_value` above); a
    focused row's affordance un-dims to (about) the value's own colour."""
    row = DetailValueRow(
        "Repeat", "Weekly", affordance=True, can_focus=True, id="row"
    )
    app = _RowHarness(row)
    async with app.run_test(size=(40, 5)) as pilot:
        # A focusable widget auto-focuses on app start -- blur it first so
        # "resting" actually means unfocused, not just "before the test's
        # own `.focus()` call".
        row.blur()
        await pilot.pause()

        affordance = row.query_one(".detail-value-row-affordance")
        value = row.query_one(".detail-value-row-value")
        value_luminance = _relative_luminance(_painted_color(app, value))
        resting_luminance = _relative_luminance(_painted_color(app, affordance))
        assert resting_luminance < value_luminance

        row.focus()
        await pilot.pause()
        focused_luminance = _relative_luminance(_painted_color(app, affordance))
        assert focused_luminance == pytest.approx(value_luminance, abs=0.02)


@pytest.mark.asyncio
async def test_a_select_editor_mounted_via_begin_edit_still_opens_on_enter():
    """Fix round 1, review finding 1 (task-1-review.md): an unconditional
    `event.stop()`/`prevent_default()` in `_on_key`'s already-editing
    branch silently ate a mounted `Select`'s own Enter-to-open binding
    (`expanded` stayed `False`, verified against a bare `Select` which
    does open). Textual only resolves a non-priority `BINDINGS` action
    (`Select`'s `Binding("enter,down,space,up", "show_overlay")`) once the
    raw `Key` event bubbles unstopped all the way up to `App` -- a
    `DetailValueRow` ancestor stopping it broke the editor `begin_edit`'s
    own docstring names as the typical case. This is the regression test
    the review said would have caught it."""
    row = DetailValueRow(
        "Repeat", "Weekly", affordance=True, can_focus=True, id="row"
    )
    app = _RowHarness(row)
    async with app.run_test(size=(40, 8)) as pilot:
        select: Select[str] = Select(
            [("Weekly", "weekly"), ("Daily", "daily")], id="editor"
        )
        row.begin_edit(select)
        await pilot.pause()
        assert app.focused is select
        assert select.expanded is False

        await pilot.press("enter")
        await pilot.pause()
        assert select.expanded is True, (
            "Select's own Enter-to-open binding must still fire while it "
            "is the row's open editor"
        )
