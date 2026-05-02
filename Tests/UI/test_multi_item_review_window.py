from types import SimpleNamespace

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Widgets.multi_item_review_window import (
    CANCEL_GENERATION_DISABLED_TOOLTIP,
    CANCEL_GENERATION_ENABLED_TOOLTIP,
    GENERATE_ANALYSES_ENABLED_TOOLTIP,
    GENERATE_ANALYSES_IN_PROGRESS_TOOLTIP,
    GENERATE_ANALYSES_NO_SELECTION_TOOLTIP,
    MultiItemReviewWindow,
)


@pytest.mark.asyncio
async def test_multi_item_review_generation_actions_explain_states():
    """Generation controls should explain selection, ready, and running states."""

    class MultiItemReviewApp(App[None]):
        def compose(self) -> ComposeResult:
            yield MultiItemReviewWindow(SimpleNamespace(notify=lambda *args, **kwargs: None))

    app = MultiItemReviewApp()
    async with app.run_test() as pilot:
        window = app.query_one(MultiItemReviewWindow)
        generate_button = window.query_one("#generate-analyses", Button)
        cancel_button = window.query_one("#cancel-generation", Button)

        # State 1: no selected media, so generation cannot start.
        window.selected_items = []
        window.analysis_in_progress = False
        window.update_generate_button()
        await pilot.pause()

        assert generate_button.disabled is True
        assert str(generate_button.tooltip) == GENERATE_ANALYSES_NO_SELECTION_TOOLTIP
        assert cancel_button.disabled is True
        assert str(cancel_button.tooltip) == CANCEL_GENERATION_DISABLED_TOOLTIP

        # State 2: selected media is ready to generate, with no cancellation target.
        window.selected_items = [{"id": 1, "title": "Transcript"}]
        window.analysis_in_progress = False
        window.update_generate_button()
        await pilot.pause()

        assert generate_button.disabled is False
        assert str(generate_button.tooltip) == GENERATE_ANALYSES_ENABLED_TOOLTIP
        assert cancel_button.disabled is True
        assert str(cancel_button.tooltip) == CANCEL_GENERATION_DISABLED_TOOLTIP

        # State 3: an active generation can be cancelled but not duplicated.
        window.analysis_in_progress = True
        window.update_generate_button()
        await pilot.pause()

        assert generate_button.disabled is True
        assert str(generate_button.tooltip) == GENERATE_ANALYSES_IN_PROGRESS_TOOLTIP
        assert cancel_button.disabled is False
        assert str(cancel_button.tooltip) == CANCEL_GENERATION_ENABLED_TOOLTIP
