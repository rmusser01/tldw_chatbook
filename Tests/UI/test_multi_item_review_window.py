from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Widgets.multi_item_review_window import MultiItemReviewWindow


@pytest.mark.asyncio
async def test_multi_item_review_generation_actions_explain_states():
    class MultiItemReviewApp(App[None]):
        def compose(self) -> ComposeResult:
            yield MultiItemReviewWindow(SimpleNamespace(notify=lambda *args, **kwargs: None))

    app = MultiItemReviewApp()
    async with app.run_test() as pilot:
        window = app.query_one(MultiItemReviewWindow)
        generate_button = window.query_one("#generate-analyses", Button)
        cancel_button = window.query_one("#cancel-generation", Button)

        window.selected_items = []
        window.analysis_in_progress = False
        window.update_generate_button()
        await pilot.pause()

        assert generate_button.disabled is True
        assert "Select at least one media item before generating analyses" in str(generate_button.tooltip)
        assert cancel_button.disabled is True
        assert "No analysis generation is running" in str(cancel_button.tooltip)

        window.selected_items = [{"id": 1, "title": "Transcript"}]
        window.analysis_in_progress = False
        window.update_generate_button()
        await pilot.pause()

        assert generate_button.disabled is False
        assert "Generate analyses for the selected media items" in str(generate_button.tooltip)
        assert cancel_button.disabled is True
        assert "No analysis generation is running" in str(cancel_button.tooltip)

        window.analysis_in_progress = True
        window.update_generate_button()
        await pilot.pause()

        assert generate_button.disabled is True
        assert "Analysis generation is already running" in str(generate_button.tooltip)
        assert cancel_button.disabled is False
        assert "Cancel the running analysis generation" in str(cancel_button.tooltip)
