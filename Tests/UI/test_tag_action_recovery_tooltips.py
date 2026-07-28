from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from textual.widgets import Button

from tldw_chatbook.Widgets.collections_tag_window import CollectionsTagWindow


@pytest.mark.parametrize(
    ("selected_keywords", "expected"),
    [
        (
            [],
            {
                "rename-keyword": (
                    True,
                    "Select exactly one keyword or tag before renaming.",
                ),
                "merge-keywords": (
                    True,
                    "Select at least two keywords or tags before merging.",
                ),
                "delete-keywords": (
                    True,
                    "Select one or more keywords or tags before deleting.",
                ),
            },
        ),
        (
            [{"id": 1, "keyword": "research", "usage_count": 3}],
            {
                "rename-keyword": (False, "Rename the selected keyword or tag."),
                "merge-keywords": (
                    True,
                    "Select at least two keywords or tags before merging.",
                ),
                "delete-keywords": (
                    False,
                    "Delete the selected keyword or tag.",
                ),
            },
        ),
        (
            [
                {"id": 1, "keyword": "research", "usage_count": 3},
                {"id": 2, "keyword": "notes", "usage_count": 5},
            ],
            {
                "rename-keyword": (
                    True,
                    "Select exactly one keyword or tag before renaming.",
                ),
                "merge-keywords": (
                    False,
                    "Merge the selected keywords or tags.",
                ),
                "delete-keywords": (
                    False,
                    "Delete the selected keywords or tags.",
                ),
            },
        ),
    ],
)
def test_tag_action_button_contracts(selected_keywords, expected):
    """Exercise the action-state function without a surrogate Textual app."""
    window = CollectionsTagWindow(
        app_instance=SimpleNamespace(media_db=None, notify=Mock())
    )
    buttons = {
        "rename-keyword": Button("Rename"),
        "merge-keywords": Button("Merge"),
        "delete-keywords": Button("Delete"),
    }
    window.query_one = Mock(
        side_effect=lambda selector, *_args: buttons[selector.removeprefix("#")]
    )
    window.selected_keywords = selected_keywords

    window.update_action_buttons()

    for button_id, (disabled, tooltip) in expected.items():
        assert buttons[button_id].disabled is disabled
        assert str(buttons[button_id].tooltip) == tooltip
