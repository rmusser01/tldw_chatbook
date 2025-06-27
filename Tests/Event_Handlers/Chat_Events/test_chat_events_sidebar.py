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
    handle_media_item_selected,
    handle_chat_media_copy_title_button_pressed,
    handle_chat_media_copy_content_button_pressed,
    handle_chat_media_copy_author_button_pressed,
    handle_chat_media_copy_url_button_pressed,
)

# Import our comprehensive mock fixture
from Tests.fixtures.event_handler_mocks import create_comprehensive_app_mock, create_widget_mock

pytestmark = pytest.mark.asyncio


@pytest.fixture
def mock_app():
    """Use the base mock app with sidebar-specific overrides."""
    # Get the base mock app from our fixtures
    app = create_comprehensive_app_mock()
    
    # Override specific widgets for sidebar tests if needed
    # (The base mock already includes all the media sidebar widgets)
    
    return app


def test_clear_and_disable_media_display(mock_app):
    """Test that all copy buttons are disabled and the current item is cleared."""
    # _clear_and_disable_media_display is synchronous, not async
    _clear_and_disable_media_display(mock_app)

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
    mock_app.media_db.search_media_db.return_value = (mock_media_items, len(mock_media_items))
    mock_results_list = mock_app.query_one("#chat-media-search-results-listview", ListView)

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar.ListItem',
               side_effect=ListItem) as mock_list_item_class:
        await perform_media_sidebar_search(mock_app, "test term")

        # ListView.clear is async in the base mock
        mock_results_list.clear.assert_called_once()
        # TextArea.clear is sync
        mock_app.query_one("#chat-media-content-display", TextArea).clear.assert_called_once()
        mock_app.media_db.search_media_db.assert_called_once()
        # ListView.append is async in the base mock
        assert mock_results_list.append.call_count == 2

        # Check that ListItem was called with a Label containing the correct text
        first_call_args = mock_list_item_class.call_args_list[0].args
        assert isinstance(first_call_args[0], Label)
        assert "Test Title 1" in first_call_args[0].renderable


async def test_perform_media_sidebar_search_no_results(mock_app):
    """Test searching with a term that returns no results."""
    mock_app.media_db.search_media_db.return_value = ([], 0)
    mock_results_list = mock_app.query_one("#chat-media-search-results-listview", ListView)

    # Track Label calls
    label_calls = []
    
    def track_label_creation(*args, **kwargs):
        label_calls.append((args, kwargs))
        return Label(*args, **kwargs)
    
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar.Label',
               side_effect=track_label_creation):
        await perform_media_sidebar_search(mock_app, "no results term")

    # ListView.append is async in the base mock
    mock_results_list.append.assert_called_once()
    
    # Check that Label was created with the correct text
    assert len(label_calls) == 1
    assert label_calls[0][0] == ("No media found.",)
    
    # The call argument is a ListItem containing a real Label
    call_arg = mock_results_list.append.call_args[0][0]
    assert isinstance(call_arg, ListItem)


async def test_perform_media_sidebar_search_empty_term(mock_app):
    """Test that an empty search term clears results and does not search."""
    await perform_media_sidebar_search(mock_app, "")
    mock_app.media_db.search_media_db.assert_not_called()
    mock_app.query_one("#chat-media-search-results-listview", ListView).clear.assert_called_once()


async def test_handle_chat_media_search_input_changed_debouncing(mock_app):
    """Test that input changes are debounced via set_timer."""
    # handle_chat_media_sidebar_input_changed doesn't take input widget as parameter
    await handle_chat_media_sidebar_input_changed(mock_app)

    # Check that set_timer was called with appropriate delay and callback
    mock_app.set_timer.assert_called_once()
    # First arg is delay, second is callback
    delay = mock_app.set_timer.call_args[0][0]
    callback = mock_app.set_timer.call_args[0][1]
    assert delay == 0.5
    assert callable(callback)


async def test_handle_chat_media_load_selected_button_pressed(mock_app):
    """Test loading a selected media item into the display."""
    # Mock the light data on the list item
    media_data_light = {
        'id': 123,
        'title': 'Loaded Title', 
        'author': 'Author Name', 
        'media_type': 'Article',
        'url': 'http://example.com'
    }
    
    # Mock the full data from DB
    media_data_full = {
        'id': 123,
        'title': 'Loaded Title', 
        'author': 'Author Name', 
        'media_type': 'Article',
        'url': 'http://example.com', 
        'content': 'This is the full content.'
    }
    
    mock_list_item = MagicMock(spec=ListItem)
    setattr(mock_list_item, 'media_data', media_data_light)

    # Mock DB to return full data
    mock_app.media_db.get_media_by_id.return_value = media_data_full

    # Call the actual function
    await handle_media_item_selected(mock_app, mock_list_item)

    # Verify DB was called with correct ID
    mock_app.media_db.get_media_by_id.assert_called_once_with(123)
    
    # Verify current_sidebar_media_item was set
    assert mock_app.current_sidebar_media_item == media_data_full
    
    # Verify TextAreas were updated with correct data
    mock_app.query_one("#chat-media-title-display", TextArea).load_text.assert_called_once_with('Loaded Title')
    mock_app.query_one("#chat-media-content-display", TextArea).load_text.assert_called_once_with('This is the full content.')
    mock_app.query_one("#chat-media-author-display", TextArea).load_text.assert_called_once_with('Author Name')
    mock_app.query_one("#chat-media-url-display", TextArea).load_text.assert_called_once_with('http://example.com')

    # Verify copy buttons were enabled
    assert mock_app.query_one("#chat-media-copy-title-button", Button).disabled is False
    assert mock_app.query_one("#chat-media-copy-content-button", Button).disabled is False
    assert mock_app.query_one("#chat-media-copy-author-button", Button).disabled is False
    assert mock_app.query_one("#chat-media-copy-url-button", Button).disabled is False


async def test_handle_chat_media_load_selected_no_selection(mock_app):
    """Test load button when nothing is selected."""
    # Create a list item without media_data attribute
    mock_list_item = MagicMock(spec=ListItem)
    # Don't set media_data attribute

    # Call the function
    await handle_media_item_selected(mock_app, mock_list_item)

    # Should notify user of the issue
    mock_app.notify.assert_called_once_with("Selected item has no data.", severity="warning")
    
    # Should not attempt to load from DB
    mock_app.media_db.get_media_by_id.assert_not_called()
    
    # The _clear_and_disable_media_display should have been called at the start
    # which clears all the text areas
    assert mock_app.query_one("#chat-media-title-display", TextArea).clear.called
    assert mock_app.query_one("#chat-media-content-display", TextArea).clear.called
    assert mock_app.query_one("#chat-media-author-display", TextArea).clear.called
    assert mock_app.query_one("#chat-media-url-display", TextArea).clear.called
    
    # Copy buttons should be disabled
    assert mock_app.query_one("#chat-media-copy-title-button", Button).disabled is True
    assert mock_app.query_one("#chat-media-copy-content-button", Button).disabled is True
    assert mock_app.query_one("#chat-media-copy-author-button", Button).disabled is True
    assert mock_app.query_one("#chat-media-copy-url-button", Button).disabled is True


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