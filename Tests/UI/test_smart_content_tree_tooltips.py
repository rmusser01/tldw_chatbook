import pytest

from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Widgets.SmartContentTree import SmartContentTree


@pytest.mark.asyncio
async def test_smart_content_tree_bulk_controls_have_descriptive_tooltips():
    """Chatbook content selection controls should explain bulk actions."""

    class SmartContentTreeApp(App):
        def compose(self) -> ComposeResult:
            yield SmartContentTree()

    app = SmartContentTreeApp()

    async with app.run_test() as pilot:
        tree = app.query_one(SmartContentTree)
        expected_tooltips = {
            "apply-filter": "Filter the content tree using the search text and enabled categories.",
            "select-all": "Select every currently visible content item.",
            "select-none": "Clear every selected content item.",
            "select-invert": "Reverse selection for the currently visible content items.",
        }

        for button_id, expected_tooltip in expected_tooltips.items():
            button = tree.query_one(f"#{button_id}", Button)
            assert str(button.tooltip) == expected_tooltip
