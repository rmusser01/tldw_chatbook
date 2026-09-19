"""Production-CSS behavior of the Inspector's retained list/detail reader."""

from pathlib import Path
from typing import ClassVar

import pytest
from textual.widgets import Button, OptionList, TextArea

from Tests.UI.consolidated_css import APP_STYLESHEETS, ConsolidatedCSSApp
from tldw_chatbook.Widgets.Console.console_inspector_detail_pane import (
    ConsoleInspectorDetailPane,
)
from tldw_chatbook.Widgets.Console.console_inspector_presentation import (
    InspectorSection,
)


class PaneHarness(ConsolidatedCSSApp):
    CSS_PATH: ClassVar[list[Path]] = list(APP_STYLESHEETS)

    def compose(self):
        yield ConsoleInspectorDetailPane(
            (
                InspectorSection("one", "Preview", "Messages"),
                InspectorSection("two", "Preview", "Tools"),
            )
        )

    def on_console_inspector_detail_pane_section_selected(self, event):
        event.pane.set_detail(
            event.key, event.key, "\n".join(str(i) for i in range(100))
        )


@pytest.mark.asyncio
async def test_keyboard_back_and_resize_preserve_selection_and_reader():
    app = PaneHarness()
    async with app.run_test(size=(120, 40)) as pilot:
        pane = app.query_one(ConsoleInspectorDetailPane)
        pane.select("two")
        await pilot.pause()
        reader = pane.query_one(TextArea)
        reader.scroll_to(y=20, animate=False)
        await pilot.pause()
        old_scroll = reader.scroll_y
        await pilot.resize_terminal(80, 24)
        await pilot.pause()
        back = pane.query_one(Button)
        assert back.display
        assert reader.is_on_screen
        assert await pilot.click(back)
        await pilot.pause()
        assert pane.query_one(OptionList).has_focus
        assert pane.selected_key == "two"
        await pilot.press("enter")
        await pilot.pause()
        assert reader.has_focus
        assert reader.text.startswith("0\n1")
        assert reader.scroll_y == old_scroll
        await pilot.resize_terminal(120, 40)
        await pilot.pause()
        assert pane.selected_key == "two"
        assert pane.query_one(OptionList).display
        assert reader.is_on_screen
