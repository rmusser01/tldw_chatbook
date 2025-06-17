# /tests/Event_Handlers/Chat_Events/test_chat_events_sidebar.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from textual.widgets import Button, Input, ListView, TextArea, ListItem, Label
from textual.css.query import QueryError

# Functions to test
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar import (
    _clear_and_disable_media_display,
    perform_media_sidebar_search,
    handle_chat_media_sidebar_input_changed,
    handle_chat_media_copy_title_button_pressed,
    handle_chat_media_copy_content_button_pressed,
    handle_chat_media_copy_author_button_pressed,
    handle_chat_media_copy_url_button_pressed,
)

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_app():
    """Provides a comprehensive mock of the TldwCli app."""
    app = AsyncMock()

    # Mock UI components
    app.query_one = MagicMock()
    mock_results_list = AsyncMock(spec=ListView)
    mock_review_display = AsyncMock(spec=TextArea)
    mock_copy_title_btn = MagicMock(spec=Button)
    mock_copy_content_btn = MagicMock(spec=Button)
    mock_copy_author_btn = MagicMock(spec=Button)
    mock_copy_url_btn = MagicMock(spec=Button)
    mock_search_input = MagicMock(spec=Input)

    # Configure query_one to return the correct mock widget
    def query_one_side_effect(selector, _type):
        if selector == "#chat-media-search-results-listview":
            return mock_results_list
        if selector == "#chat-media-content-display":
            return mock_review_display
        if selector == "#chat-media-copy-title-button":
            return mock_copy_title_btn
        if selector == "#chat-media-copy-content-button":
            return mock_copy_content_btn
        if selector == "#chat-media-copy-author-button":
            return mock_copy_author_btn
        if selector == "#chat-media-copy-url-button":
            return mock_copy_url_btn
        if selector == "#chat-media-search-input":
            return mock_search_input
        raise QueryError(f"Widget not found: {selector}")

    app.query_one.side_effect = query_one_side_effect

    # Mock DB and state
    app.media_db = MagicMock()
    app.current_sidebar_media_item = None

    # Mock app methods
    app.notify = AsyncMock()
    app.copy_to_clipboard = MagicMock()
    app.set_timer = MagicMock()
    app.run_worker = MagicMock()

    # For debouncing timer
    app._media_sidebar_search_timer = None

    return app


async def test_disable_media_copy_buttons(mock_app):
    """Test that all copy buttons are disabled and the current item is cleared."""
    await _disable_media_copy_buttons(mock_app)

    assert mock_app.current_sidebar_media_item is None
    assert mock_app.query_one("#chat-media-copy-title-button", Button).disabled is True
    assert mock_app.query_one("#chat-media-copy-content-button", Button).disabled is True
    assert mock_app.query_one("#chat-media-copy-author-button", Button).disabled is True
    assert mock_app.query_one("#chat-media-copy-url-button", Button).disabled is True


async def test_perform_media_sidebar_search_with_results(mock_app):
    """Test searching with a term that returns results."""
    mock_media_items = [
        {'title': 'Test Title 1', 'media_id': 'id12345678'},
        {'title': 'Test Title 2', 'media_id': 'id87654321'},
    ]
    mock_app.media_db.search_media_db.return_value = mock_media_items
    mock_results_list = mock_app.query_one("#chat-media-search-results-listview", ListView)

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar.ListItem',
               side_effect=ListItem) as mock_list_item_class:
        await perform_media_sidebar_search(mock_app, "test term")

        mock_results_list.clear.assert_called_once()
        mock_app.query_one("#chat-media-content-display", TextArea).clear.assert_called_once()
        mock_app.media_db.search_media_db.assert_called_once()
        assert mock_results_list.append.call_count == 2

        # Check that ListItem was called with a Label containing the correct text
        first_call_args = mock_list_item_class.call_args_list[0].args
        assert isinstance(first_call_args[0], Label)
        assert "Test Title 1" in first_call_args[0].renderable


async def test_perform_media_sidebar_search_no_results(mock_app):
    """Test searching with a term that returns no results."""
    mock_app.media_db.search_media_db.return_value = []
    mock_results_list = mock_app.query_one("#chat-media-search-results-listview", ListView)

    await perform_media_sidebar_search(mock_app, "no results term")

    mock_results_list.append.assert_called_once()
    # The call argument is a ListItem, which contains a Label. We check the Label's content.
    call_arg = mock_results_list.append.call_args[0][0]
    assert isinstance(call_arg, ListItem)
    assert call_arg.children[0].renderable == "No media found."


async def test_perform_media_sidebar_search_empty_term(mock_app):
    """Test that an empty search term clears results and does not search."""
    await perform_media_sidebar_search(mock_app, "")
    mock_app.media_db.search_media_db.assert_not_called()
    mock_app.query_one("#chat-media-search-results-listview", ListView).clear.assert_called_once()


async def test_handle_chat_media_search_input_changed_debouncing(mock_app):
    """Test that input changes are debounced via set_timer."""
    mock_timer = MagicMock()
    mock_app._media_sidebar_search_timer = mock_timer
    mock_input_widget = MagicMock(spec=Input, value=" new search ")

    await handle_chat_media_search_input_changed(mock_app, mock_input_widget)

    mock_timer.stop.assert_called_once()
    mock_app.set_timer.assert_called_once()
    # Check that run_worker is part of the callback, which calls perform_media_sidebar_search
    callback_lambda = mock_app.set_timer.call_args[0][1]
    # We can't easily execute the lambda here, but we can verify it's set.
    assert callable(callback_lambda)


async def test_handle_chat_media_load_selected_button_pressed(mock_app):
    """Test loading a selected media item into the display."""
    media_data = {
        'title': 'Loaded Title', 'author': 'Author Name', 'media_type': 'Article',
        'url': 'http://example.com', 'content': 'This is the full content.'
    }
    mock_list_item = MagicMock(spec=ListItem)
    mock_list_item.media_data = media_data

    mock_results_list = mock_app.query_one("#chat-media-search-results-listview", ListView)
    mock_results_list.highlighted_child = mock_list_item

    await handle_chat_media_load_selected_button_pressed(mock_app, MagicMock())

    assert mock_app.current_sidebar_media_item == media_data
    mock_app.query_one("#chat-media-content-display", TextArea).load_text.assert_called_once()
    loaded_text = mock_app.query_one("#chat-media-content-display", TextArea).load_text.call_args[0][0]
    assert "Title: Loaded Title" in loaded_text
    assert "Author: Author Name" in loaded_text
    assert "This is the full content." in loaded_text

    assert mock_app.query_one("#chat-media-copy-title-button", Button).disabled is False


async def test_handle_chat_media_load_selected_no_selection(mock_app):
    """Test load button when nothing is selected."""
    mock_results_list = mock_app.query_one("#chat-media-search-results-listview", ListView)
    mock_results_list.highlighted_child = None

    await handle_chat_media_load_selected_button_pressed(mock_app, MagicMock())

    mock_app.notify.assert_called_with("No media item selected.", severity="warning")
    mock_app.query_one("#chat-media-content-display", TextArea).clear.assert_called_once()
    assert mock_app.query_one("#chat-media-copy-title-button", Button).disabled is True


async def test_handle_copy_buttons_with_data(mock_app):
    """Test all copy buttons when data is available."""
    media_data = {'title': 'Copy Title', 'content': 'Copy Content', 'author': 'Copy Author', 'url': 'http://copy.url'}
    mock_app.current_sidebar_media_item = media_data

    # Test copy title
    await handle_chat_media_copy_title_button_pressed(mock_app, MagicMock())
    mock_app.copy_to_clipboard.assert_called_with('Copy Title')
    mock_app.notify.assert_called_with("Title copied to clipboard.")

    # Test copy content
    await handle_chat_media_copy_content_button_pressed(mock_app, MagicMock())
    mock_app.copy_to_clipboard.assert_called_with('Copy Content')
    mock_app.notify.assert_called_with("Content copied to clipboard.")

    # Test copy author
    await handle_chat_media_copy_author_button_pressed(mock_app, MagicMock())
    mock_app.copy_to_clipboard.assert_called_with('Copy Author')
    mock_app.notify.assert_called_with("Author copied to clipboard.")

    # Test copy URL
    await handle_chat_media_copy_url_button_pressed(mock_app, MagicMock())
    mock_app.copy_to_clipboard.assert_called_with('http://copy.url')
    mock_app.notify.assert_called_with("URL copied to clipboard.")


async def test_handle_copy_buttons_no_data(mock_app):
    """Test copy buttons when data is not available."""
    mock_app.current_sidebar_media_item = None

    # Test copy title
    await handle_chat_media_copy_title_button_pressed(mock_app, MagicMock())
    mock_app.notify.assert_called_with("No media title to copy.", severity="warning")

    # Test copy content
    await handle_chat_media_copy_content_button_pressed(mock_app, MagicMock())
    mock_app.notify.assert_called_with("No media content to copy.", severity="warning")

    # Test copy author
    await handle_chat_media_copy_author_button_pressed(mock_app, MagicMock())
    mock_app.notify.assert_called_with("No media author to copy.", severity="warning")

    # Test copy URL
    await handle_chat_media_copy_url_button_pressed(mock_app, MagicMock())
    mock_app.notify.assert_called_with("No media URL to copy.", severity="warning")