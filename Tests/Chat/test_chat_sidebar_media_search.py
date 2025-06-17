import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

from textual.app import App
from textual.widgets import Button, TextArea, Static, Select, Checkbox, Input, Label, ListView, ListItem, Collapsible
from textual.containers import VerticalScroll
from rich.text import Text  # Keep if ChatMessage or other components use it directly

# Mocked app class (simplified)
from tldw_chatbook.app import TldwCli
# Assuming ccl might be needed if any underlying app logic touches it, otherwise can be removed if not directly relevant
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase  # For type hinting if needed

# Event handlers to be tested (or called directly in tests)
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar import (
    _clear_and_disable_media_display,
    perform_media_sidebar_search,
    handle_chat_media_copy_title_button_pressed,
    handle_chat_media_copy_content_button_pressed,
    handle_chat_media_copy_author_button_pressed,
    handle_chat_media_copy_url_button_pressed
)


@pytest_asyncio.fixture
async def mock_app_media_test(tmp_path):  # Renamed fixture for clarity
    """A mock TldwCli app instance tailored for media search sidebar tests."""
    default_config_content = """
[general]
log_level = "DEBUG"
default_tab = "chat"
USERS_NAME = "TestUser"
[chat_defaults]
provider = "Ollama" # Minimal provider needed if LLM calls were involved
model = "test_model"
"""
    config_path = tmp_path / "test_config_media.toml"
    with open(config_path, "w") as f:
        f.write(default_config_content)

    with patch('tldw_chatbook.config.DEFAULT_CONFIG_PATH', new=config_path):
        app = TldwCli()
        app.API_IMPORTS_SUCCESSFUL = True

        # Mock essential UI components that are queried globally or by these handlers
        def side_effect_query_one(selector, *args, **kwargs):
            # print(f"DEBUG: query_one called with selector: {selector}") # For debugging tests
            if selector == "#chat-media-search-collapsible":
                return MagicMock(spec=Collapsible)
            elif selector == "#chat-media-search-input":
                return MagicMock(spec=Input, value="")
            elif selector == "#chat-media-search-results-listview":
                mock_list_view = MagicMock(spec=ListView)
                mock_list_view._nodes = []
                type(mock_list_view).children = PropertyMock(side_effect=lambda: mock_list_view._nodes)
                mock_list_view.clear = MagicMock(side_effect=lambda: setattr(mock_list_view, '_nodes', []))

                def append_item(item):
                    if not isinstance(item, ListItem):
                        li = ListItem(Label(str(getattr(item, 'label', str(item)))))
                        if hasattr(item, 'media_data'):
                            li.media_data = item.media_data
                        mock_list_view._nodes.append(li)
                    else:
                        mock_list_view._nodes.append(item)

                mock_list_view.append = MagicMock(side_effect=append_item)
                mock_list_view.highlighted_child = None
                return mock_list_view
            elif selector == "#chat-media-load-selected-button":
                return MagicMock(spec=Button)
            elif selector == "#chat-media-review-display":
                mock_text_area = MagicMock(spec=TextArea)
                # Use a variable to store the text for the PropertyMock
                _text_storage = ""

                def set_text_storage(value):
                    nonlocal _text_storage
                    _text_storage = value

                def get_text_storage():
                    nonlocal _text_storage
                    return _text_storage

                type(mock_text_area).text = PropertyMock(fget=get_text_storage, fset=set_text_storage)
                mock_text_area.clear = MagicMock(side_effect=lambda: set_text_storage(""))
                mock_text_area.load_text = MagicMock(side_effect=lambda t: set_text_storage(t))
                set_text_storage("")  # Initialize
                return mock_text_area
            elif selector == "#chat-media-copy-title-button":
                return MagicMock(spec=Button, disabled=True)
            elif selector == "#chat-media-copy-content-button":
                return MagicMock(spec=Button, disabled=True)
            elif selector == "#chat-media-copy-author-button":
                return MagicMock(spec=Button, disabled=True)
            elif selector == "#chat-media-copy-url-button":
                return MagicMock(spec=Button, disabled=True)
            # Add fallbacks for other critical UI elements if handlers touch them
            elif selector == "#chat-input":  # Example if some handler indirectly queries it
                return MagicMock(spec=TextArea, text="")
            else:
                print(f"Warning: mock_app_media_test query_one unmocked: {selector}")
                return MagicMock()

        app.query_one = MagicMock(side_effect=side_effect_query_one)

        app.notify = MagicMock()
        app.copy_to_clipboard = MagicMock()

        # Mock DB and services relevant to media search
        app.notes_service = MagicMock()
        mock_media_db_instance = MagicMock(spec=MediaDatabase)  # Use spec for better mocking
        app.notes_service._get_db = MagicMock(return_value=mock_media_db_instance)
        app.mock_media_db_instance = mock_media_db_instance  # For direct access in tests

        # Initialize app attributes related to this feature
        app.current_sidebar_media_item = None
        app._media_sidebar_search_timer = None
        app.notes_user_id = "test_user"  # Default user_id for tests
        app._ui_ready = True  # Simulate UI is ready

        yield app

        # Teardown
        app._ui_ready = False
        app.current_sidebar_media_item = None
        if hasattr(app, '_media_sidebar_search_timer') and app._media_sidebar_search_timer:
            app._media_sidebar_search_timer.cancel()
            app._media_sidebar_search_timer = None


# --- Test Data ---
MOCK_MEDIA_ITEM_1 = {
    'media_id': 'media123', 'uuid': 'uuid123', 'title': 'Test Media One', 'content': 'Content for one.',
    'author': 'Author One', 'url': 'http://example.com/one', 'media_type': 'article', 'keywords': ['test', 'one'],
    'notes': 'Notes for one', 'publication_date': '2023-01-01'
}
MOCK_MEDIA_ITEM_2 = {
    'media_id': 'media456', 'uuid': 'uuid456', 'title': 'Test Media Two', 'content': 'Content for two.',
    'author': 'Author Two', 'url': 'http://example.com/two', 'media_type': 'video', 'keywords': ['test', 'two'],
    'notes': 'Notes for two', 'publication_date': '2023-01-02'
}
MOCK_MEDIA_SEARCH_RESULTS = [MOCK_MEDIA_ITEM_1, MOCK_MEDIA_ITEM_2]


# --- Test Cases ---

@pytest.mark.asyncio
async def test_media_search_initial_state(mock_app_media_test: TldwCli):
    app = mock_app_media_test

    assert app.query_one("#chat-media-search-collapsible", Collapsible) is not None
    assert app.query_one("#chat-media-search-input", Input) is not None
    assert app.query_one("#chat-media-search-results-listview", ListView) is not None
    assert app.query_one("#chat-media-load-selected-button", Button) is not None

    review_display_mock = app.query_one("#chat-media-review-display", TextArea)
    assert review_display_mock is not None
    assert type(review_display_mock).text.fget(review_display_mock) == ""

    assert app.query_one("#chat-media-copy-title-button", Button).disabled is True
    assert app.query_one("#chat-media-copy-content-button", Button).disabled is True
    assert app.query_one("#chat-media-copy-author-button", Button).disabled is True
    assert app.query_one("#chat-media-copy-url-button", Button).disabled is True
    assert app.current_sidebar_media_item is None


@pytest.mark.asyncio
async def test_media_search_functionality(mock_app_media_test: TldwCli, pilot):  # pilot for timers if used
    app = mock_app_media_test
    app.mock_media_db_instance.search_media_db = MagicMock(return_value=MOCK_MEDIA_SEARCH_RESULTS)

    search_input_mock = app.query_one("#chat-media-search-input", Input)
    results_list_view_mock = app.query_one("#chat-media-search-results-listview", ListView)

    search_input_mock.value = "test"
    perform_media_sidebar_search(app, "test")  # Call handler directly, bypassing debounce for simplicity

    app.mock_media_db_instance.search_media_db.assert_called_once_with(
        search_term="test",
        search_fields=['title', 'content', 'author', 'keywords', 'notes'],
        media_types=None,
        include_trash=False,
        include_deleted=False,
        limit=50
    )
    assert len(results_list_view_mock._nodes) == 2

    item1_widget_mock = results_list_view_mock._nodes[0]
    assert isinstance(item1_widget_mock, ListItem)
    label_widget1 = item1_widget_mock.query_one(Label)
    assert label_widget1.renderable == f"{MOCK_MEDIA_ITEM_1['title']} (ID: {MOCK_MEDIA_ITEM_1['media_id']})"
    assert item1_widget_mock.media_data == MOCK_MEDIA_ITEM_1

    # Test no results
    app.mock_media_db_instance.search_media_db.reset_mock()
    app.mock_media_db_instance.search_media_db.return_value = []
    search_input_mock.value = "nomatch"
    perform_media_sidebar_search(app, "nomatch")

    app.mock_media_db_instance.search_media_db.assert_called_once_with(
        search_term="nomatch",
        search_fields=['title', 'content', 'author', 'keywords', 'notes'],
        media_types=None,
        include_trash=False,
        include_deleted=False,
        limit=50
    )
    assert len(results_list_view_mock._nodes) == 1
    no_results_label_mock = results_list_view_mock._nodes[0].query_one(Label)
    assert no_results_label_mock.renderable == "No media found."


@pytest.mark.asyncio
async def test_media_load_for_review(mock_app_media_test: TldwCli, pilot):
    app = mock_app_media_test

    app.mock_media_db_instance.search_media_db = MagicMock(return_value=[MOCK_MEDIA_ITEM_1])
    perform_media_sidebar_search(app, "item1")

    results_list_view_mock = app.query_one("#chat-media-search-results-listview", ListView)
    assert len(results_list_view_mock._nodes) == 1
    results_list_view_mock.highlighted_child = results_list_view_mock._nodes[0]

    app.mock_media_db_instance.get_media_by_id = MagicMock(return_value=MOCK_MEDIA_ITEM_1)
    review_display_mock = app.query_one("#chat-media-review-display", TextArea)

    await handle_chat_media_load_selected_button_pressed(app)

    app.mock_media_db_instance.get_media_by_id.assert_called_once_with(MOCK_MEDIA_ITEM_1['media_id'])

    expected_review_text_parts = [
        f"Title: {MOCK_MEDIA_ITEM_1['title']}",
        f"Author: {MOCK_MEDIA_ITEM_1['author']}",
        f"Type: {MOCK_MEDIA_ITEM_1['media_type']}",
        f"URL: {MOCK_MEDIA_ITEM_1['url']}",
        f"Date: {MOCK_MEDIA_ITEM_1['publication_date']}",
        "\n--- Content Snippet ---\n",
        MOCK_MEDIA_ITEM_1['content']
    ]
    expected_review_text = "\n".join(expected_review_text_parts)
    assert type(review_display_mock).text.fget(review_display_mock) == expected_review_text

    assert app.current_sidebar_media_item == MOCK_MEDIA_ITEM_1
    assert app.query_one("#chat-media-copy-title-button", Button).disabled is False
    assert app.query_one("#chat-media-copy-content-button", Button).disabled is False
    assert app.query_one("#chat-media-copy-author-button", Button).disabled is False
    assert app.query_one("#chat-media-copy-url-button", Button).disabled is False


@pytest.mark.asyncio
@pytest.mark.parametrize("button_id, field_key, expected_notification", [
    ("chat-media-copy-title-button", "title", "Title copied to clipboard."),
    ("chat-media-copy-content-button", "content", "Content copied to clipboard."),
    ("chat-media-copy-author-button", "author", "Author copied to clipboard."),
    ("chat-media-copy-url-button", "url", "URL copied to clipboard."),
])
async def test_media_copy_buttons(mock_app_media_test: TldwCli, pilot, button_id, field_key, expected_notification):
    app = mock_app_media_test

    handler_map = {
        "chat-media-copy-title-button": handle_chat_media_copy_title_button_pressed,
        "chat-media-copy-content-button": handle_chat_media_copy_content_button_pressed,
        "chat-media-copy-author-button": handle_chat_media_copy_author_button_pressed,
        "chat-media-copy-url-button": handle_chat_media_copy_url_button_pressed,
    }
    copy_handler = handler_map[button_id]

    app.current_sidebar_media_item = MOCK_MEDIA_ITEM_1
    app.query_one(f"#{button_id}", Button).disabled = False

    app.copy_to_clipboard.reset_mock()
    app.notify.reset_mock()

    await copy_handler(app)

    app.copy_to_clipboard.assert_called_once_with(str(MOCK_MEDIA_ITEM_1[field_key]))
    app.notify.assert_called_with(expected_notification)


@pytest.mark.asyncio
async def test_media_review_clearing_on_new_empty_search(mock_app_media_test: TldwCli, pilot):
    app = mock_app_media_test

    app.current_sidebar_media_item = MOCK_MEDIA_ITEM_1
    review_display_mock = app.query_one("#chat-media-review-display", TextArea)
    type(review_display_mock).text.fset(review_display_mock, "Some loaded text")

    # Simulate buttons being enabled
    app.query_one("#chat-media-copy-title-button", Button).disabled = False
    app.query_one("#chat-media-copy-content-button", Button).disabled = False
    app.query_one("#chat-media-copy-author-button", Button).disabled = False
    app.query_one("#chat-media-copy-url-button", Button).disabled = False

    app.mock_media_db_instance.search_media_db = MagicMock(return_value=[])
    perform_media_sidebar_search(app, "newsearchterm_that_returns_nothing")

    assert type(review_display_mock).text.fget(review_display_mock) == ""
    assert app.current_sidebar_media_item is None
    assert app.query_one("#chat-media-copy-title-button", Button).disabled is True
    assert app.query_one("#chat-media-copy-content-button", Button).disabled is True
    assert app.query_one("#chat-media-copy-author-button", Button).disabled is True
    assert app.query_one("#chat-media-copy-url-button", Button).disabled is True

    # Example of testing the debounced search input if needed (more complex)
    # @pytest.mark.asyncio
    # async def test_media_search_input_debounced(mock_app_media_test: TldwCli, pilot):
    #     app = mock_app_media_test
    #     app.mock_media_db_instance.search_media_db = MagicMock(return_value=MOCK_MEDIA_SEARCH_RESULTS)
    #     search_input_mock = app.query_one("#chat-media-search-input", Input)

    #     from tldw_chatbook.Event_Handlers.Chat_Events.chat_events_sidebar import handle_chat_media_search_input_changed

    #     search_input_mock.value = "debounced_test"
    #     # Instead of calling perform_media_sidebar_search directly, call the input handler
    #     await handle_chat_media_search_input_changed(app, search_input_mock)

    #     # Assert DB not called immediately
    #     app.mock_media_db_instance.search_media_db.assert_not_called()

    #     await pilot.pause(0.6) # Wait for debounce timer (default 0.5s in handler)

    #     app.mock_media_db_instance.search_media_db.assert_called_once_with(
    #         search_term="debounced_test",
    #         search_fields=['title', 'content', 'author', 'keywords', 'notes'],
    #         media_types=None,
    #         include_trash=False,
    #         include_deleted=False,
    #         limit=50
    #     )
    #     results_list_view_mock = app.query_one("#chat-media-search-results-listview", ListView)
    #     assert len(results_list_view_mock._nodes) == 2

# End of test_chat_sidebar_media_search.py
