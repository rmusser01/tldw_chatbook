import pytest

from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.UI.Wizards.EmbeddingSteps import SmartContentSelector


@pytest.mark.asyncio
async def test_embedding_wizard_content_selector_bulk_controls_have_tooltips():
    """Embedding source selection controls should clarify their bulk scope."""

    class SmartContentSelectorApp(App):
        def compose(self) -> ComposeResult:
            yield SmartContentSelector("custom")

    app = SmartContentSelectorApp()

    async with app.run_test() as pilot:
        expected_tooltips = {
            "select-all": "Select every visible source item for this embedding collection.",
            "clear-all": "Clear every selected source item.",
            "invert-selection": "Reverse selection for the visible source items.",
        }

        for button_id, expected_tooltip in expected_tooltips.items():
            button = app.query_one(f"#{button_id}", Button)
            assert str(button.tooltip) == expected_tooltip
