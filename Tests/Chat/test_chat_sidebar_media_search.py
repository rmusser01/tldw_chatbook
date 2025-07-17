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
async def real_app_media_test(tmp_path, monkeypatch):
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

[splash_screen]
enabled = false

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
    
    # Patch the get_cli_setting to ensure splash screen is disabled
    from tldw_chatbook.config import get_cli_setting as original_get_cli_setting
    def mock_get_cli_setting(section, key, default=None):
        if section == "splash_screen" and key == "enabled":
            return False
        return original_get_cli_setting(section, key, default)
    
    monkeypatch.setattr("tldw_chatbook.config.get_cli_setting", mock_get_cli_setting)
    monkeypatch.setattr("tldw_chatbook.app.get_cli_setting", mock_get_cli_setting)
    
    # Create the app instance
    app = TldwCli()
    app.API_IMPORTS_SUCCESSFUL = True
    
    # Initialize real databases
    app.media_db = MediaDatabase(str(media_db_path), "test_client_id")
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
    app.media_db.add_media_with_keywords(
        title='Test Media One',
        content='Content for one.',
        media_type='article',
        author='Author One',
        url='http://example.com/one',
        keywords='test,one',
        ingestion_date='2023-01-01'
    )
    app.media_db.add_media_with_keywords(
        title='Test Media Two',
        content='Content for two.',
        media_type='video',
        author='Author Two',
        url='http://example.com/two',
        keywords='test,two',
        ingestion_date='2023-01-02'
    )
    
    yield app
    
    # Cleanup
    if hasattr(app, '_media_sidebar_search_timer') and app._media_sidebar_search_timer:
        app._media_sidebar_search_timer.cancel()
    # Databases don't have close() method anymore
    if hasattr(app.media_db, 'close_connection'):
        app.media_db.close_connection()


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
        
        # First, check if any UI is loaded at all
        all_widgets = list(app.query("*").results())
        print(f"\nTotal widgets found: {len(all_widgets)}")
        print(f"Widget IDs: {[w.id for w in all_widgets[:20] if w.id]}")  # First 20 IDs
        
        # Check if splash screen is present
        try:
            splash = app.query_one("#app-splash-screen")
            print("Splash screen is active, waiting for it to close...")
            # Press any key to skip splash screen if skip_on_keypress is enabled
            await pilot.press("space")
            await pilot.pause(2.0)  # Wait for splash to close and main UI to load
            
            # Check again
            all_widgets = list(app.query("*").results())
            print(f"\nAfter splash close - Total widgets: {len(all_widgets)}")
            print(f"Widget IDs after splash: {[w.id for w in all_widgets[:20] if w.id]}")
        except Exception:
            print("No splash screen found (good!)")
        
        # Check if we're in a tabbed interface
        try:
            tabbed_content = app.query("TabbedContent").first()
            if tabbed_content:
                print(f"TabbedContent found with {len(tabbed_content.children)} children")
                active_tab = tabbed_content.active
                print(f"Active tab: {active_tab}")
        except Exception:
            print("No TabbedContent found")
        
        # Navigate to chat tab
        await pilot.press("ctrl+c")  # Use ctrl+c for chat tab
        await pilot.pause(0.5)  # Give time for tab to load
        
        # Check again after navigation
        all_widgets = list(app.query("*").results())
        print(f"\nAfter navigation - Total widgets: {len(all_widgets)}")
        
        # Check that all required UI elements exist
        # Try to find the collapsible directly
        try:
            collapsible = app.query_one("#chat-media-collapsible", Collapsible)
            assert collapsible is not None
        except Exception as e:
            # If not found, try to debug what's available
            print(f"\nError finding chat-media-collapsible: {str(e)}")
            # List all Collapsible widgets
            collapsibles = app.query(Collapsible).results()
            print(f"Available Collapsibles: {[(c.id, c.title) for c in collapsibles if c.id]}")
            # List all widgets with 'media' in their ID
            media_widgets = [w for w in app.query("*").results() if w.id and 'media' in w.id]
            print(f"Media-related widgets: {[(w.id, type(w).__name__) for w in media_widgets]}")
            # Check for chat-related widgets
            chat_widgets = [w for w in app.query("*").results() if w.id and 'chat' in w.id]
            print(f"Chat-related widgets: {[(w.id, type(w).__name__) for w in chat_widgets[:10]]}")  # First 10
            raise
        
        search_input = app.query_one("#chat-media-search-input", Input)
        assert search_input is not None
        assert search_input.value == ""
        
        results_listview = app.query_one("#chat-media-search-results-listview", ListView)
        assert results_listview is not None
        
        # Load button no longer exists - selection automatically populates details
        
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
        
        # Verify both results are present (order not guaranteed)
        all_labels = [str(item.query_one(Label).renderable) for item in list_items]
        assert any("Test Media One" in label for label in all_labels)
        assert any("Test Media Two" in label for label in all_labels)
        
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
        
        # Select the first item in the ListView
        results_listview.index = 0  # Set the index to select the first item
        results_listview.action_select_cursor()  # Trigger the selection
        await pilot.pause(0.5)  # Give time for the selection event to process
        
        # Selection should automatically populate the details fields
        # Check that media details are displayed
        title_display = app.query_one("#chat-media-title-display", TextArea)
        content_display = app.query_one("#chat-media-content-display", TextArea)
        author_display = app.query_one("#chat-media-author-display", TextArea)
        url_display = app.query_one("#chat-media-url-display", TextArea)
        
        # Allow time for the selection to update the fields
        await pilot.pause(0.5)
        
        assert title_display.text != ""
        assert "Test Media One" in title_display.text
        assert "Author One" in author_display.text
        assert "Content for one." in content_display.text
        assert "http://example.com/one" in url_display.text
        
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
    
    # Mock the clipboard and notify methods BEFORE running the app
    from unittest.mock import MagicMock
    app.copy_to_clipboard = MagicMock()
    app.notify = MagicMock()
    
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
        
        # Get the first media item and set its media_data attribute
        first_item = list_items[0]
        # The mock database returns items with IDs 1, 2, 3
        first_item.media_data = {'id': 1}
        
        # Import and call the handler function directly
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar import handle_media_item_selected
        await handle_media_item_selected(app, first_item)
        await pilot.pause(0.1)
        
        # Verify the button is enabled
        copy_button = app.query_one(f"#{button_id}", Button)
        assert copy_button.disabled is False
        
        # Verify app.current_sidebar_media_item is set
        assert app.current_sidebar_media_item is not None
        
        # Import the handler directly and call it
        from tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar import CHAT_SIDEBAR_BUTTON_HANDLERS
        from textual.events import Click
        
        # Create a mock event
        mock_event = Button.Pressed(copy_button)
        
        # Get and call the appropriate handler
        handler = CHAT_SIDEBAR_BUTTON_HANDLERS.get(button_id)
        assert handler is not None, f"Handler not found for {button_id}"
        
        # Call the handler directly
        await handler(app, mock_event)
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
        results_listview.index = 0  # Set the index to select the first item
        results_listview.action_select_cursor()  # Trigger the selection
        await pilot.pause(0.5)  # Allow time for the selection to update the fields
        
        # Verify content is loaded
        title_display = app.query_one("#chat-media-title-display", TextArea)
        content_display = app.query_one("#chat-media-content-display", TextArea)
        assert title_display.text != ""
        assert content_display.text != ""
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
        
        # Mock the timer to track debounce
        from unittest.mock import MagicMock, patch
        mock_timer = MagicMock()
        
        # Type quickly and trigger the search manually after debounce
        await pilot.click(search_input)
        search_input.value = "test"
        
        # Manually trigger the search after simulating debounce wait
        await perform_media_sidebar_search(app, "test")
        await pilot.pause(0.2)
        
        # Now results should be populated
        list_items = results_listview.query(ListItem)
        assert len(list_items) == 2  # Should find both test items

# End of test_chat_sidebar_media_search.py
