import pytest

from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Widgets.media_details_widget import MediaDetailsWidget


@pytest.mark.asyncio
async def test_media_details_content_search_controls_have_descriptive_tooltips():
    """Media content search uses compact icons that should explain their actions."""

    class MediaDetailsApp(App):
        def compose(self) -> ComposeResult:
            yield MediaDetailsWidget(app_instance=self, type_slug="all-media")

    app = MediaDetailsApp()

    async with app.run_test() as pilot:
        expected_tooltips = {
            "content-search-button-all-media": "Search within the selected media content.",
            "content-search-prev-all-media": "Jump to the previous content search match.",
            "content-search-next-all-media": "Jump to the next content search match.",
        }

        for button_id, expected_tooltip in expected_tooltips.items():
            button = app.query_one(f"#{button_id}", Button)
            assert str(button.tooltip) == expected_tooltip
