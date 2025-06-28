import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path

from textual.widgets import Button, TextArea, Input, Label, ListView, ListItem, Collapsible
from rich.text import Text

# Real app class for integration testing
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB

# Event handlers to be tested
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar import (
    _clear_and_disable_media_display,
    perform_media_sidebar_search,
    handle_chat_media_copy_title_button_pressed,
    handle_chat_media_copy_content_button_pressed,
    handle_chat_media_copy_author_button_pressed,
    handle_chat_media_copy_url_button_pressed
)


@pytest_asyncio.fixture
async def real_app_media_test(tmp_path):
    """A real TldwCli app instance for integration testing media search sidebar."""
    # Create temporary directories for databases
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    media_db_path = db_dir / "media.db"
    chachanotes_db_path = db_dir / "chachanotes.db"
    
    # Create config file
    default_config_content = """
[general]
log_level = "DEBUG"
default_tab = "chat"
USERS_NAME = "TestUser"

[chat_defaults]
provider = "Ollama"
model = "test_model"

[paths]
data_dir = "{}"
""".format(str(tmp_path))
    
    config_path = tmp_path / "test_config_media.toml"
    with open(config_path, "w") as f:
        f.write(default_config_content)

    # Set environment variables for test
    os.environ['TLDW_CONFIG_PATH'] = str(config_path)
    
    # Create the app instance
    app = TldwCli()
    app.API_IMPORTS_SUCCESSFUL = True
    
    # Initialize real databases
    app.media_db = MediaDatabase(str(media_db_path))
    app.chachanotes_db = CharactersRAGDB(str(chachanotes_db_path), "test_user")
    
    # Set up the notes service to use real database
    if hasattr(app, 'notes_service'):
        app.notes_service._db = app.media_db
        app.notes_service._get_db = lambda: app.media_db
    
    # Initialize app attributes
    app.current_sidebar_media_item = None
    app._media_sidebar_search_timer = None
    app.notes_user_id = "test_user"
    
    # Insert test data into media database
    app.media_db.insert_media_item(
        title='Test Media One',
        content='Content for one.',
        media_type='article',
        author='Author One',
        url='http://example.com/one',
        keywords=['test', 'one'],
        notes='Notes for one',
        publication_date='2023-01-01'
    )
    app.media_db.insert_media_item(
        title='Test Media Two',
        content='Content for two.',
        media_type='video',
        author='Author Two',
        url='http://example.com/two',
        keywords=['test', 'two'],
        notes='Notes for two',
        publication_date='2023-01-02'
    )
    
    yield app
    
    # Cleanup
    if hasattr(app, '_media_sidebar_search_timer') and app._media_sidebar_search_timer:
        app._media_sidebar_search_timer.cancel()
    app.media_db.close()
    app.chachanotes_db.close()


# Test marker for integration tests
pytestmark = pytest.mark.integration


# --- Test Cases ---

@pytest.mark.asyncio
async def test_media_search_initial_state(real_app_media_test: TldwCli):
    """Test that media search sidebar has correct initial state."""
    app = real_app_media_test
    
    async with app.run_test() as pilot:
        # Wait for UI to be ready
        await pilot.pause(0.5)
        
        # Navigate to chat tab
        await pilot.press("c")
        await pilot.pause(0.1)
        
        # Check that all required UI elements exist
        collapsible = app.query_one("#chat-media-search-collapsible", Collapsible)
        assert collapsible is not None
        
        search_input = app.query_one("#chat-media-search-input", Input)
        assert search_input is not None
        assert search_input.value == ""
        
        results_listview = app.query_one("#chat-media-search-results-listview", ListView)
        assert results_listview is not None
        
        load_button = app.query_one("#chat-media-load-selected-button", Button)
        assert load_button is not None
        
        # Check copy buttons are initially disabled
        copy_title_btn = app.query_one("#chat-media-copy-title-button", Button)
        copy_content_btn = app.query_one("#chat-media-copy-content-button", Button)
        copy_author_btn = app.query_one("#chat-media-copy-author-button", Button)
        copy_url_btn = app.query_one("#chat-media-copy-url-button", Button)
        
        assert copy_title_btn.disabled is True
        assert copy_content_btn.disabled is True
        assert copy_author_btn.disabled is True
        assert copy_url_btn.disabled is True
        
        assert app.current_sidebar_media_item is None


@pytest.mark.asyncio
async def test_media_search_functionality(real_app_media_test: TldwCli):
    """Test media search functionality with real database."""
    app = real_app_media_test
    
    async with app.run_test() as pilot:
        # Wait for UI to be ready
        await pilot.pause(0.5)
        
        # Navigate to chat tab
        await pilot.press("c")
        await pilot.pause(0.1)
        
        # Get UI elements
        search_input = app.query_one("#chat-media-search-input", Input)
        results_listview = app.query_one("#chat-media-search-results-listview", ListView)
        
        # Clear any existing search results
        await results_listview.clear()
        
        # Type search query
        await pilot.click(search_input)
        await pilot.pause(0.1)
        search_input.value = "test"
        
        # Trigger search directly (bypassing debounce)
        await perform_media_sidebar_search(app, "test")
        await pilot.pause(0.2)
        
        # Check results
        list_items = results_listview.query(ListItem)
        assert len(list_items) == 2  # Should find both test items
        
        # Verify first result
        first_item = list_items[0]
        label = first_item.query_one(Label)
        assert "Test Media One" in str(label.renderable)
        
        # Test search with no results
        await results_listview.clear()
        search_input.value = "nomatch"
        await perform_media_sidebar_search(app, "nomatch")
        await pilot.pause(0.2)
        
        # Check no results message
        list_items = results_listview.query(ListItem)
        assert len(list_items) == 1
        no_results_label = list_items[0].query_one(Label)
        assert "No media found" in str(no_results_label.renderable)


@pytest.mark.asyncio
async def test_media_load_for_review(real_app_media_test: TldwCli):
    """Test loading media item for review."""
    app = real_app_media_test
    
    async with app.run_test() as pilot:
        # Wait for UI to be ready
        await pilot.pause(0.5)
        
        # Navigate to chat tab
        await pilot.press("c")
        await pilot.pause(0.1)
        
        # Search for first item
        search_input = app.query_one("#chat-media-search-input", Input)
        await pilot.click(search_input)
        search_input.value = "One"
        await perform_media_sidebar_search(app, "One")
        await pilot.pause(0.2)
        
        # Get results and select first item
        results_listview = app.query_one("#chat-media-search-results-listview", ListView)
        list_items = results_listview.query(ListItem)
        assert len(list_items) == 1
        
        # Click on the first item to highlight it
        await pilot.click(list_items[0])
        await pilot.pause(0.1)
        
        # Click load button
        load_button = app.query_one("#chat-media-load-selected-button", Button)
        await pilot.click(load_button)
        await pilot.pause(0.2)
        
        # Check that media was loaded
        review_display = app.query_one("#chat-media-review-display", TextArea)
        assert review_display.text != ""
        assert "Test Media One" in review_display.text
        assert "Author One" in review_display.text
        assert "Content for one." in review_display.text
        
        # Check copy buttons are now enabled
        assert app.query_one("#chat-media-copy-title-button", Button).disabled is False
        assert app.query_one("#chat-media-copy-content-button", Button).disabled is False
        assert app.query_one("#chat-media-copy-author-button", Button).disabled is False
        assert app.query_one("#chat-media-copy-url-button", Button).disabled is False
        
        # Verify app state
        assert app.current_sidebar_media_item is not None
        assert app.current_sidebar_media_item['title'] == 'Test Media One'


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id, field_key, expected_notification", [
    ("chat-media-copy-title-button", "title", "Title copied to clipboard."),
    ("chat-media-copy-content-button", "content", "Content copied to clipboard."),
    ("chat-media-copy-author-button", "author", "Author copied to clipboard."),
    ("chat-media-copy-url-button", "url", "URL copied to clipboard."),
])
async def test_media_copy_buttons(real_app_media_test: TldwCli, button_id, field_key, expected_notification):
    """Test copy buttons functionality."""
    app = real_app_media_test
    
    async with app.run_test() as pilot:
        # Wait for UI to be ready
        await pilot.pause(0.5)
        
        # Navigate to chat tab
        await pilot.press("c")
        await pilot.pause(0.1)
        
        # First load a media item
        search_input = app.query_one("#chat-media-search-input", Input)
        await pilot.click(search_input)
        search_input.value = "One"
        await perform_media_sidebar_search(app, "One")
        await pilot.pause(0.2)
        
        # Select and load first item
        results_listview = app.query_one("#chat-media-search-results-listview", ListView)
        list_items = results_listview.query(ListItem)
        await pilot.click(list_items[0])
        await pilot.pause(0.1)
        
        load_button = app.query_one("#chat-media-load-selected-button", Button)
        await pilot.click(load_button)
        await pilot.pause(0.2)
        
        # Click the copy button
        copy_button = app.query_one(f"#{button_id}", Button)
        assert copy_button.disabled is False
        
        # Mock the clipboard and notify methods to verify they're called
        from unittest.mock import MagicMock
        app.copy_to_clipboard = MagicMock()
        app.notify = MagicMock()
        
        await pilot.click(copy_button)
        await pilot.pause(0.1)
        
        # Verify clipboard and notification
        expected_value = {
            'title': 'Test Media One',
            'content': 'Content for one.',
            'author': 'Author One',
            'url': 'http://example.com/one'
        }[field_key]
        
        app.copy_to_clipboard.assert_called_once_with(expected_value)
        app.notify.assert_called_with(expected_notification)


@pytest.mark.asyncio
async def test_media_review_clearing_on_new_empty_search(real_app_media_test: TldwCli):
    """Test that media review is cleared when performing new search with no results."""
    app = real_app_media_test
    
    async with app.run_test() as pilot:
        # Wait for UI to be ready
        await pilot.pause(0.5)
        
        # Navigate to chat tab
        await pilot.press("c")
        await pilot.pause(0.1)
        
        # First load a media item
        search_input = app.query_one("#chat-media-search-input", Input)
        await pilot.click(search_input)
        search_input.value = "One"
        await perform_media_sidebar_search(app, "One")
        await pilot.pause(0.2)
        
        # Select and load first item
        results_listview = app.query_one("#chat-media-search-results-listview", ListView)
        list_items = results_listview.query(ListItem)
        await pilot.click(list_items[0])
        await pilot.pause(0.1)
        
        load_button = app.query_one("#chat-media-load-selected-button", Button)
        await pilot.click(load_button)
        await pilot.pause(0.2)
        
        # Verify content is loaded
        review_display = app.query_one("#chat-media-review-display", TextArea)
        assert review_display.text != ""
        assert app.current_sidebar_media_item is not None
        
        # Verify copy buttons are enabled
        assert app.query_one("#chat-media-copy-title-button", Button).disabled is False
        
        # Now search for something that returns no results
        search_input.clear()
        search_input.value = "nonexistent_search_term"
        await perform_media_sidebar_search(app, "nonexistent_search_term")
        await pilot.pause(0.2)
        
        # Check that display areas are cleared
        title_display = app.query_one("#chat-media-title-display", TextArea)
        content_display = app.query_one("#chat-media-content-display", TextArea)
        author_display = app.query_one("#chat-media-author-display", TextArea)
        url_display = app.query_one("#chat-media-url-display", TextArea)
        
        assert title_display.text == ""
        assert content_display.text == ""
        assert author_display.text == ""
        assert url_display.text == ""
        
        # Check app state is cleared
        assert app.current_sidebar_media_item is None
        
        # Check copy buttons are disabled again
        assert app.query_one("#chat-media-copy-title-button", Button).disabled is True
        assert app.query_one("#chat-media-copy-content-button", Button).disabled is True
        assert app.query_one("#chat-media-copy-author-button", Button).disabled is True
        assert app.query_one("#chat-media-copy-url-button", Button).disabled is True


@pytest.mark.asyncio
async def test_media_search_input_debounced(real_app_media_test: TldwCli):
    """Test that search input is debounced."""
    app = real_app_media_test
    
    async with app.run_test() as pilot:
        # Wait for UI to be ready
        await pilot.pause(0.5)
        
        # Navigate to chat tab
        await pilot.press("c")
        await pilot.pause(0.1)
        
        # Get search input
        search_input = app.query_one("#chat-media-search-input", Input)
        results_listview = app.query_one("#chat-media-search-results-listview", ListView)
        
        # Type quickly
        await pilot.click(search_input)
        await pilot.type("t")
        await pilot.pause(0.1)
        await pilot.type("e")
        await pilot.pause(0.1)
        await pilot.type("s")
        await pilot.pause(0.1)
        await pilot.type("t")
        
        # Check that results aren't updated immediately
        list_items = results_listview.query(ListItem)
        assert len(list_items) == 0  # No results yet due to debounce
        
        # Wait for debounce timer (typically 0.5s)
        await pilot.pause(0.6)
        
        # Now results should be populated
        list_items = results_listview.query(ListItem)
        assert len(list_items) == 2  # Should find both test items

# End of test_chat_sidebar_media_search.py
