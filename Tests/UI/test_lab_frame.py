"""The shared Lab frame: regions, status row, lazy body, and collapse."""

from __future__ import annotations

import pytest
from textual.app import ComposeResult
from textual.widget import Widget
from textual.widgets import Static

from tldw_chatbook.UI.Lab_Modules.lab_rail_layout import LAB_RAIL_INSPECTOR
from tldw_chatbook.UI.Screens.lab_frame import LabScreen, LabStatusChip
from tldw_chatbook.UI.Workbench.workbench_state import WorkbenchHeaderState
from Tests.UI.test_screen_navigation import _build_test_app

class _ProbeBody(Static):
    """Stands in for a mode's expensive legacy window."""


class _ProbeLabScreen(LabScreen):
    """A minimal Lab mode used to exercise the frame itself."""

    def __init__(self, app_instance, *, chips=(), **kwargs):
        super().__init__(app_instance, "llm", **kwargs)
        self._chips = chips
        self.body_ready_calls = 0

    def lab_header_state(self) -> WorkbenchHeaderState:
        return WorkbenchHeaderState(title="Probe", subtitle="probe mode")

    def lab_status_chips(self) -> tuple[LabStatusChip, ...]:
        return self._chips

    def compose_lab_rail(self) -> ComposeResult:
        yield Static("rail row", id="probe-rail-row")

    def build_lab_body(self) -> Widget:
        return _ProbeBody("body", id="probe-body")

    def on_lab_body_ready(self) -> None:
        self.body_ready_calls += 1


def _mount(screen_factory):
    """Build the test app and this probe screen.

    Deliberately does NOT load the CSS bundle: every assertion in this file
    is about behaviour and class membership, not rendered styling. Setting
    `app.CSS_PATH` after construction would be worse than useless anyway --
    `App.__init__` reads `css_path or self.CSS_PATH` once, at construction,
    so a post-hoc assignment silently does nothing. Styling assertions live
    in test_lab_workbench.py, which uses a class-level CSS_PATH.
    """
    app = _build_test_app()
    return app, screen_factory(app)


@pytest.mark.asyncio
async def test_the_body_is_absent_at_first_paint_and_present_after_deferral():
    """The lazy mount is the whole performance claim -- assert it directly.

    Without this, a frame that mounted the body inline would pass every
    other test in this file.
    """
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        assert not screen.query(_ProbeBody), "body mounted during first paint"
        await pilot.pause()
        await pilot.pause()
        assert screen.query_one(_ProbeBody) is not None
        assert screen.body_ready_calls == 1


@pytest.mark.asyncio
async def test_a_mode_with_no_chips_renders_no_status_row_at_all():
    """A mode without status must not reserve a row of dead chrome.

    Models always supplies a chip, so this path has no real consumer until
    Speech and Evals adopt -- it would otherwise ship unexercised.
    """
    app, screen = _mount(lambda a: _ProbeLabScreen(a, chips=()))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert not screen.query("#lab-status-row")


@pytest.mark.asyncio
async def test_chips_render_and_refresh_mutates_them_without_recomposing():
    """Refresh must update the same Static, not replace the row."""
    chips = [LabStatusChip(chip_id="servers", text="Servers: none running")]

    class _Screen(_ProbeLabScreen):
        def lab_status_chips(self):
            return tuple(chips)

    app, screen = _mount(lambda a: _Screen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        chip_widget = screen.query_one("#lab-status-chip-servers", Static)
        assert "Servers: none running" in str(chip_widget.renderable)

        chips[0] = LabStatusChip(chip_id="servers", text="Servers: 2 running")
        screen.refresh_lab_status()
        await pilot.pause()

        assert screen.query_one("#lab-status-chip-servers", Static) is chip_widget
        assert "Servers: 2 running" in str(chip_widget.renderable)


@pytest.mark.asyncio
async def test_the_frame_wires_its_route_into_the_mode_strip():
    """The existing strip suite mounts the strip standalone and would not
    notice the frame passing the wrong active_route."""
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        active = screen.query_one("#lab-mode-models")
        assert "is-active" in active.classes


@pytest.mark.asyncio
async def test_the_inspector_starts_collapsed_on_first_run():
    app, screen = _mount(lambda a: _ProbeLabScreen(a))
    async with app.run_test() as pilot:
        await app.push_screen(screen)
        await pilot.pause()
        assert screen.rail_layout.is_collapsed(LAB_RAIL_INSPECTOR) is True
        assert screen.query_one("#lab-inspector-handle") is not None
