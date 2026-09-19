"""Frame state transitions under the production stylesheet cascade."""

import pytest
from textual.app import App, ComposeResult
from textual.containers import Vertical
from textual.widgets import Button, Static

from Tests.UI.consolidated_css import BUNDLED_STYLESHEET
from tldw_chatbook.UI.Console_Modules.frame import (
    frame_console_region,
    sync_console_focus_paint,
)


class FrameHost(App):
    CSS_PATH = str(BUNDLED_STYLESHEET)

    def compose(self) -> ComposeResult:
        with Vertical(id="console-left-rail"):
            yield Button("Context", id="console-context-rail-collapse")
        yield Static("Other", id="other")

    @staticmethod
    def _is_descendant_or_self(focused, region):
        return focused is region or focused is not None and region in focused.ancestors


@pytest.mark.asyncio
async def test_frame_edge_quiet_and_focus_transitions() -> None:
    async with FrameHost().run_test() as pilot:
        region = pilot.app.query_one("#console-left-rail")
        button = region.query_one(Button)
        frame_console_region(region, edges=("right",))
        await pilot.pause()
        assert region.styles.border_right[0] == "solid"
        assert region.styles.border_left[0] == ""
        normal = region.styles.border_right[1]
        sync_console_focus_paint(pilot.app, button)
        await pilot.pause()
        assert region.styles.border_right[1] != normal
        sync_console_focus_paint(pilot.app, None)
        await pilot.pause()
        assert region.styles.border_right[1] == normal
        frame_console_region(region, variant="quiet")
        await pilot.pause()
        assert all(
            getattr(region.styles, f"border_{edge}")[0] == ""
            for edge in ("top", "right", "bottom", "left")
        )
        frame_console_region(region, edges=("left", "bottom"))
        await pilot.pause()
        assert region.styles.border_right[0] == ""
        assert region.styles.border_left[0] == "solid"
        assert region.styles.border_bottom[0] == "solid"
        assert not region.has_class("console-frame-quiet")
        left_normal = region.styles.border_left[1]
        sync_console_focus_paint(pilot.app, button)
        await pilot.pause()
        assert region.styles.border_left[1] != left_normal
        assert region.styles.border_right[0] == ""
        frame_console_region(region, variant="quiet")
        sync_console_focus_paint(pilot.app, button)
        await pilot.pause()
        assert region.styles.border_left[0] == ""
        frame_console_region(region, top=False, bottom=False)
        await pilot.pause()
        assert region.styles.border_top[0] == ""
        assert region.styles.border_bottom[0] == ""
        assert region.styles.border_left[0] == "solid"
        assert region.styles.border_right[0] == "solid"
