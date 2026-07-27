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
    LAB_INSPECTOR_WIDTH,
    LAB_RAIL_ROW_CLASS,
    LAB_RAIL_WIDTH,
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

    def __init__(self, layout: LabRailLayout) -> None:
        super().__init__()
        self._layout = layout

    def compose(self) -> ComposeResult:
        yield LabWorkbench(rail_layout=self._layout, id="lab-workbench")

    def on_mount(self) -> None:
        rail = self.query_one("#lab-rail")
        for index, name in enumerate(("Llama.cpp", "Llamafile", "Ollama")):
            row = Button(name, id=f"lab-rail-row-{index}", classes=LAB_RAIL_ROW_CLASS)
            if index == 1:
                row.add_class("is-active")
            rail.mount(row)
        self.query_one("#lab-body").mount(Static("body", id="probe-body"))


@pytest.mark.asyncio
async def test_all_three_regions_render_when_nothing_is_collapsed():
    app = _WorkbenchHarness(LabRailLayout())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        assert app.query_one("#lab-rail").display is True
        assert app.query_one("#lab-body").display is True
        assert app.query_one("#lab-inspector").display is True
        assert not app.query("#lab-rail-handle")
        assert not app.query("#lab-inspector-handle")


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
        assert rail.width == LAB_RAIL_WIDTH
        assert handle.width == 11
        assert body.width >= 63
        assert rail.width + body.width + handle.width <= 100


@pytest.mark.asyncio
async def test_the_selected_rail_row_gets_no_border_and_stays_one_row_high():
    """The bundle's global `.is-active` rule must not reach rail rows.

    At height 1 an unneutralised `is-active` row renders region.height == 2 --
    a half-bordered artifact that displaces its neighbours. Asserting the
    border alone would miss a height regression, so assert both.
    """
    app = _WorkbenchHarness(LabRailLayout())
    async with app.run_test(size=(120, 30)) as pilot:
        await pilot.pause()
        rows = [app.query_one(f"#lab-rail-row-{i}", Button) for i in range(3)]
        active = rows[1]
        assert "is-active" in active.classes

        border = active.styles.border
        assert not any(
            edge[0] for edge in (border.top, border.right, border.bottom, border.left)
        ), "selected rail row has a border; it will displace its neighbours"
        assert {row.region.height for row in rows} == {1}
