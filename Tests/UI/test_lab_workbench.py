"""Geometry and selection styling for the Lab frame's workbench container."""

from __future__ import annotations

from pathlib import Path

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button, Static

from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import (
    LAB_RAIL_INSPECTOR,
    LAB_RAIL_LEFT,
    LabRailLayout,
)
from tldw_chatbook.UI.Lab_Modules.lab_workbench import (
    LAB_RAIL_ROW_CLASS,
    LAB_RAIL_MAX_WIDTH,
    LAB_RAIL_MIN_WIDTH,
    LabWorkbench,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BUNDLED_STYLESHEET = _REPO_ROOT / "tldw_chatbook/css/tldw_cli_modular.tcss"


class _WorkbenchHarness(App[None]):
    """Mount the workbench with the production stylesheet.

    The bundle is required: the selection-border defect under test lives in
    the bundle's global `.is-active` rule, which beats DEFAULT_CSS. A harness
    without CSS_PATH would pass vacuously.
    """

    CSS_PATH = str(_BUNDLED_STYLESHEET)

    def __init__(
        self,
        layout: LabRailLayout,
        row_labels: tuple[str, ...] = ("Llama.cpp", "Llamafile", "Ollama"),
    ) -> None:
        super().__init__()
        self._layout = layout
        # Index 1 is the one marked active below, so a caller probing the
        # narrowest case should put its longest label there: the active row
        # is a column narrower than its siblings (it spends one on the
        # accent bar), which makes it the worst case for truncation.
        self._row_labels = row_labels

    def compose(self) -> ComposeResult:
        yield LabWorkbench(rail_layout=self._layout, id="lab-workbench")

    def on_mount(self) -> None:
        rail = self.query_one("#lab-rail")
        for index, name in enumerate(self._row_labels):
            row = Button(name, id=f"lab-rail-row-{index}", classes=LAB_RAIL_ROW_CLASS)
            if index == 1:
                row.add_class("is-active")
            rail.mount(row)
        self.query_one("#lab-body").mount(Static("body", id="probe-body"))


@pytest.mark.asyncio
async def test_all_three_regions_render_when_nothing_is_collapsed():
    """The handles are always composed now (task-5 fix); assert hidden, not
    absent -- ``apply_rail_layout`` toggles ``display`` in place so a rail
    toggle never has to remount the regions the frame already populated."""
    app = _WorkbenchHarness(LabRailLayout())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail").display is True
        assert app.query_one("#lab-body").display is True
        assert app.query_one("#lab-inspector").display is True
        assert app.query_one("#lab-rail-handle").display is False
        assert app.query_one("#lab-inspector-handle").display is False


@pytest.mark.asyncio
async def test_a_collapsed_rail_is_replaced_by_its_handle():
    app = _WorkbenchHarness(LabRailLayout(collapsed=frozenset({LAB_RAIL_LEFT})))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail").display is False
        assert app.query_one("#lab-rail-handle").display is True


@pytest.mark.asyncio
async def test_a_collapsed_inspector_is_replaced_by_its_handle():
    app = _WorkbenchHarness(LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR})))
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-inspector").display is False
        assert app.query_one("#lab-inspector-handle").display is True


@pytest.mark.asyncio
async def test_the_hundred_column_width_contract_holds():
    """Rail + body + collapsed inspector handle must fit 100 columns.

    Both rails open at 100 is explicitly NOT guaranteed, matching Console.
    """
    app = _WorkbenchHarness(LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR})))
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        rail = app.query_one("#lab-rail").region
        body = app.query_one("#lab-body").region
        handle = app.query_one("#lab-inspector-handle").region
        assert LAB_RAIL_MIN_WIDTH <= rail.width <= LAB_RAIL_MAX_WIDTH
        assert handle.width == 11
        assert body.width >= 63
        assert rail.width + body.width + handle.width <= 100


@pytest.mark.asyncio
async def test_the_rail_scales_with_the_terminal_instead_of_staying_fixed():
    """The rail must respond to terminal width, clamped to its bounds.

    A fixed-width rail measured 26 cells at 80, 120 and 200 columns alike --
    a third of an 80-column terminal for a list of short labels. Asserting
    the bounds alone would still pass for a constant, so this asserts the
    narrow terminal actually yields a narrower rail than the wide one.
    """
    widths: dict[int, int] = {}
    for columns in (80, 200):
        # A fresh App per run: a Textual App instance is not re-runnable.
        app = _WorkbenchHarness(
            LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR}))
        )
        async with app.run_test(size=(columns, 30)) as pilot:
            await pilot.pause()
            widths[columns] = app.query_one("#lab-rail").region.width

    for columns, width in widths.items():
        assert LAB_RAIL_MIN_WIDTH <= width <= LAB_RAIL_MAX_WIDTH, (
            f"rail width {width} out of bounds at {columns} columns"
        )
    assert widths[80] < widths[200], (
        f"rail did not scale with the terminal: {widths}"
    )


@pytest.mark.asyncio
async def test_no_rail_label_is_truncated_at_the_narrowest_supported_terminal():
    """The rail's minimum width must actually fit the longest label.

    Textual drops the whole overflowing word rather than clipping mid-word,
    so truncation is silent and total: at `min-width: 20` the 15-character
    "Download Models" rendered as "Download", with no ellipsis to hint that
    anything was lost.

    `content_region.width` is NOT a usable oracle here -- it reported 16 for
    a 15-character label that did not render. `render_line(0).text` is the
    composited output and does show the loss, so assert against that.
    """
    longest = "Download Models"
    app = _WorkbenchHarness(
        LabRailLayout(collapsed=frozenset({LAB_RAIL_INSPECTOR})),
        row_labels=("Llama.cpp", longest, "Ollama"),
    )
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        rendered = {
            str(row.label): row.render_line(0).text
            for row in app.query(f".{LAB_RAIL_ROW_CLASS}").results(Button)
        }

    for label, line in rendered.items():
        assert label in line, (
            f"rail label {label!r} was truncated to {line.strip()!r} at 80 columns"
        )


@pytest.mark.asyncio
async def test_the_selected_rail_row_gets_no_horizontal_border_and_stays_one_row_high():
    """The bundle's global `.is-active` rule must not reach rail rows.

    That rule is `border: round $ds-action-focus` -- it sets every edge. At
    height 1 the top and bottom edges each need a line the row does not
    have, so an unneutralised row renders region.height == 2: a
    half-bordered artifact that displaces its neighbours.

    The left edge is deliberately exempt. `_lab.tcss` gives the active row
    `border-left: thick $accent` as its selection marker, which consumes a
    column and never a row. Asserting "no border at all" would forbid that
    marker while catching nothing extra, so this asserts the edges that can
    actually cost a row, plus the resulting height.
    """
    app = _WorkbenchHarness(LabRailLayout())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        rows = [app.query_one(f"#lab-rail-row-{i}", Button) for i in range(3)]
        active = rows[1]
        assert "is-active" in active.classes

        border = active.styles.border
        assert not any(
            edge[0] for edge in (border.top, border.bottom)
        ), "selected rail row has a horizontal border; it will displace its neighbours"
        assert {row.region.height for row in rows} == {1}
