from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.app import App, ComposeResult
from textual.widgets import Button

from tldw_chatbook.Widgets.collections_tag_window import CollectionsTagWindow


def _assert_button_tooltips(root, expected_tooltips: dict[str, str]) -> None:
    for button_id, expected_tooltip in expected_tooltips.items():
        button = root.query_one(f"#{button_id}", Button)
        assert str(button.tooltip) == expected_tooltip


@pytest.mark.asyncio
async def test_tag_action_buttons_explain_selection_requirements_and_enabled_actions():
    app_instance = SimpleNamespace(media_db=None, notify=Mock())

    class TagWindowApp(App):
        def compose(self) -> ComposeResult:
            yield CollectionsTagWindow(app_instance=app_instance)

    app = TagWindowApp()

    async with app.run_test() as pilot:
        await pilot.pause()
        window = app.query_one(CollectionsTagWindow)

        _assert_button_tooltips(
            window,
            {
                "rename-keyword": "Select exactly one keyword or tag before renaming.",
                "merge-keywords": "Select at least two keywords or tags before merging.",
                "delete-keywords": "Select one or more keywords or tags before deleting.",
            },
        )

        window.selected_keywords = [
            {"id": 1, "keyword": "research", "usage_count": 3},
        ]
        window.update_action_buttons()

        _assert_button_tooltips(
            window,
            {
                "rename-keyword": "Rename the selected keyword or tag.",
                "merge-keywords": "Select at least two keywords or tags before merging.",
                "delete-keywords": "Delete the selected keyword or tag.",
            },
        )

        window.selected_keywords = [
            {"id": 1, "keyword": "research", "usage_count": 3},
            {"id": 2, "keyword": "notes", "usage_count": 5},
        ]
        window.update_action_buttons()

        _assert_button_tooltips(
            window,
            {
                "rename-keyword": "Select exactly one keyword or tag before renaming.",
                "merge-keywords": "Merge the selected keywords or tags.",
                "delete-keywords": "Delete the selected keywords or tags.",
            },
        )
