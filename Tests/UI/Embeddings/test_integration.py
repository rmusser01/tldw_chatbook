"""Integration coverage for the current embeddings creation surface."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from textual.widgets import Button, Checkbox, Input, Select, Tree

from Tests.textual_test_utils import widget_pilot
from tldw_chatbook.UI.SearchEmbeddingsWindow import SearchEmbeddingsWindow


@pytest.fixture
def mock_app_instance() -> MagicMock:
    app = MagicMock()
    app.notify = MagicMock()
    app.media_db = MagicMock()
    return app


@pytest.mark.asyncio
async def test_embeddings_creation_surface_mounts_current_controls(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(SearchEmbeddingsWindow, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget

        assert window.query_one("#model-select", Select)
        assert window.query_one("#collection-name", Input)
        assert window.query_one("#create-embeddings", Button)
        assert window.query_one("#content-type-chats", Checkbox)
        assert window.query_one("#content-tree", Tree)


@pytest.mark.asyncio
async def test_content_type_selection_populates_current_content_tree(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    async with await widget_pilot(SearchEmbeddingsWindow, app_instance=mock_app_instance) as pilot:
        window = pilot.app.test_widget
        chats_checkbox = window.query_one("#content-type-chats", Checkbox)

        chats_checkbox.toggle()
        await pilot.pause(0.3)

        tree = window.query_one("#content-tree", Tree)

        assert "chats" in window.selected_content_types
        assert len(tree.root.children) >= 1


@pytest.mark.asyncio
async def test_embeddings_validation_requires_collection_and_selected_content(
    mock_app_instance: MagicMock,
    widget_pilot,
) -> None:
    with patch("tldw_chatbook.UI.SearchEmbeddingsWindow.embeddings_available", True):
        async with await widget_pilot(SearchEmbeddingsWindow, app_instance=mock_app_instance) as pilot:
            window = pilot.app.test_widget

            assert window.validate_form() is False

            window.collection_name = "search-index"
            assert window.validate_form() is False

            window.selected_items = {"notes": {"note-1"}}
            assert window.validate_form() is True
