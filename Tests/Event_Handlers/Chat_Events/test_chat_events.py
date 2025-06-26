# /tests/Event_Handlers/Chat_Events/test_chat_events.py

import pytest
from unittest.mock import AsyncMock, MagicMock, patch, call

from rich.text import Text
# Mock Textual UI elements before they are imported by the module under test
from textual.widgets import (
    Button, Input, TextArea, Static, Select, Checkbox, ListView, ListItem, Label
)
from textual.containers import VerticalScroll
from textual.css.query import QueryError

# Mock DB Errors
from tldw_chatbook.DB.ChaChaNotes_DB import ConflictError, CharactersRAGDBError, InputError

# Functions to test
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import (
    handle_chat_send_button_pressed,
    handle_chat_action_button_pressed,
    handle_chat_new_conversation_button_pressed,
    handle_chat_save_current_chat_button_pressed,
    handle_chat_load_character_button_pressed,
    handle_chat_clear_active_character_button_pressed,
    # ... import other handlers as you write tests for them
)
from tldw_chatbook.Widgets.chat_message import ChatMessage

pytestmark = pytest.mark.asyncio


# A very comprehensive mock app fixture is needed here
@pytest.fixture
def mock_app():
    app = AsyncMock()
    
    # Add thread lock for chat state management
    app._chat_state_lock = MagicMock()

    # Mock services and DBs
    app.chachanotes_db = MagicMock()
    app.notes_service = MagicMock()
    app.notes_service._get_db.return_value = app.chachanotes_db
    app.media_db = MagicMock()

    # Mock core app properties
    app.API_IMPORTS_SUCCESSFUL = True
    app.app_config = {
        "api_settings": {
            "openai": {"streaming": True, "api_key_env_var": "OPENAI_API_KEY"},
            "anthropic": {"streaming": False, "api_key": "xyz-key"}
        },
        "chat_defaults": {"system_prompt": "Default system prompt."},
        "USERS_NAME": "Tester"
    }

    # Mock app state
    app.current_chat_conversation_id = None
    app.current_chat_is_ephemeral = True
    app.current_chat_active_character_data = None
    app.current_ai_message_widget = None

    # Mock app methods
    app.query_one = MagicMock()
    app.notify = AsyncMock()
    app.copy_to_clipboard = MagicMock()
    app.set_timer = MagicMock()
    app.run_worker = MagicMock()
    app.chat_wrapper = AsyncMock()
    
    # Thread-safe methods are synchronous
    app.get_current_ai_message_widget = MagicMock(return_value=None)
    app.set_current_ai_message_widget = MagicMock()

    # Timers
    app._conversation_search_timer = None

    # --- Set up mock widgets ---
    # This is complex; a helper function simplifies it.
    def setup_mock_widgets(q_one_mock):
        # Create chat log with proper async mount method
        mock_chat_log = MagicMock(spec=VerticalScroll, is_mounted=True)
        mock_chat_log.mount = AsyncMock()  # mount is async
        mock_chat_log.remove_children = AsyncMock()  # remove_children is async
        mock_chat_log.query = MagicMock(return_value=[])
        
        widgets = {
            "#chat-input": MagicMock(spec=TextArea, text="User message", is_mounted=True),
            "#chat-log": mock_chat_log,
            "#chat-api-provider": MagicMock(spec=Select, value="OpenAI"),
            "#chat-api-model": MagicMock(spec=Select, value="gpt-4"),
            "#chat-system-prompt": MagicMock(spec=TextArea, text="UI system prompt"),
            "#chat-temperature": MagicMock(spec=Input, value="0.7"),
            "#chat-top-p": MagicMock(spec=Input, value="0.9"),
            "#chat-min-p": MagicMock(spec=Input, value="0.1"),
            "#chat-top-k": MagicMock(spec=Input, value="40"),
            "#chat-llm-max-tokens": MagicMock(spec=Input, value="1024"),
            "#chat-llm-seed": MagicMock(spec=Input, value=""),
            "#chat-llm-stop": MagicMock(spec=Input, value=""),
            "#chat-llm-response-format": MagicMock(spec=Select, value="text"),
            "#chat-llm-n": MagicMock(spec=Input, value="1"),
            "#chat-llm-user-identifier": MagicMock(spec=Input, value=""),
            "#chat-llm-logprobs": MagicMock(spec=Checkbox, value=False),
            "#chat-llm-top-logprobs": MagicMock(spec=Input, value=""),
            "#chat-llm-logit-bias": MagicMock(spec=TextArea, text="{}"),
            "#chat-llm-presence-penalty": MagicMock(spec=Input, value="0.0"),
            "#chat-llm-frequency-penalty": MagicMock(spec=Input, value="0.0"),
            "#chat-llm-tools": MagicMock(spec=TextArea, text="[]"),
            "#chat-llm-tool-choice": MagicMock(spec=Input, value=""),
            "#chat-llm-fixed-tokens-kobold": MagicMock(spec=Checkbox, value=False),
            "#chat-strip-thinking-tags-checkbox": MagicMock(spec=Checkbox, value=True),
            "#chat-character-search-results-list": AsyncMock(spec=ListView),
            "#chat-character-name-edit": MagicMock(spec=Input),
            "#chat-character-description-edit": MagicMock(spec=TextArea),
            "#chat-character-personality-edit": MagicMock(spec=TextArea),
            "#chat-character-scenario-edit": MagicMock(spec=TextArea),
            "#chat-character-system-prompt-edit": MagicMock(spec=TextArea),
            "#chat-character-first-message-edit": MagicMock(spec=TextArea),
            "#chat-right-sidebar": MagicMock(),  # Mock container
        }

        def query_one_side_effect(selector, _type=None):
            # Special case for querying within the sidebar
            if isinstance(selector, MagicMock) and hasattr(selector, 'query_one'):
                return selector.query_one(selector, _type)

            if selector in widgets:
                return widgets[selector]

            # Allow querying for sub-widgets inside a container like the right sidebar
            if widgets["#chat-right-sidebar"].query_one.call_args:
                inner_selector = widgets["#chat-right-sidebar"].query_one.call_args[0][0]
                if inner_selector in widgets:
                    return widgets[inner_selector]

            raise QueryError(f"Widget not found by mock: {selector}")

        q_one_mock.side_effect = query_one_side_effect

        # Make the sidebar mock also use the main query_one logic
        widgets["#chat-right-sidebar"].query_one.side_effect = lambda sel, _type: widgets[sel]

    setup_mock_widgets(app.query_one)

    return app


# Mock external dependencies used in chat_events.py
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ccl')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.os')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessage', new_callable=AsyncMock)
async def test_handle_chat_send_button_pressed_basic(mock_chat_message_class, mock_os, mock_ccl, mock_app):
    """Test a basic message send operation."""
    mock_os.environ.get.return_value = "fake-key"

    await handle_chat_send_button_pressed(mock_app, MagicMock())

    # Assert UI updates
    mock_app.query_one("#chat-input").clear.assert_called_once()
    # Check that mount was called (it's async, so just check call count)
    assert mock_app.query_one("#chat-log").mount.call_count == 2  # User message and AI placeholder

    # Assert worker is called
    mock_app.run_worker.assert_called_once()

    # Assert chat_wrapper is called with correct parameters by the worker
    worker_lambda = mock_app.run_worker.call_args[0][0]
    worker_lambda()  # Execute the lambda to trigger the call to chat_wrapper

    mock_app.chat_wrapper.assert_called_once()
    wrapper_kwargs = mock_app.chat_wrapper.call_args.kwargs
    assert wrapper_kwargs['message'] == "User message"
    assert wrapper_kwargs['api_endpoint'] == "OpenAI"
    assert wrapper_kwargs['api_key'] == "fake-key"
    assert wrapper_kwargs['system_message'] == "UI system prompt"
    assert wrapper_kwargs['streaming'] is True  # From config


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ccl')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.os')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessage', new_callable=AsyncMock)
async def test_handle_chat_send_with_active_character(mock_chat_message_class, mock_os, mock_ccl, mock_app):
    """Test that an active character's system prompt overrides the UI."""
    mock_os.environ.get.return_value = "fake-key"
    mock_app.current_chat_active_character_data = {
        'name': 'TestChar',
        'system_prompt': 'You are TestChar.'
    }

    await handle_chat_send_button_pressed(mock_app, MagicMock())

    worker_lambda = mock_app.run_worker.call_args[0][0]
    worker_lambda()

    wrapper_kwargs = mock_app.chat_wrapper.call_args.kwargs
    assert wrapper_kwargs['system_message'] == "You are TestChar."


async def test_handle_new_conversation_button_pressed(mock_app):
    """Test that the new chat button clears state and UI."""
    # Set some state to ensure it's cleared
    mock_app.current_chat_conversation_id = "conv_123"
    mock_app.current_chat_is_ephemeral = False
    mock_app.current_chat_active_character_data = {'name': 'char'}

    await handle_chat_new_conversation_button_pressed(mock_app, MagicMock())

    mock_app.query_one("#chat-log").remove_children.assert_called_once()
    assert mock_app.current_chat_conversation_id is None
    assert mock_app.current_chat_is_ephemeral is True
    assert mock_app.current_chat_active_character_data is None
    # Check that a UI field was reset
    assert mock_app.query_one("#chat-system-prompt").text == "Default system prompt."


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ccl')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.display_conversation_in_chat_tab_ui',
       new_callable=AsyncMock)
async def test_handle_save_current_chat_button_pressed(mock_display_conv, mock_ccl, mock_app):
    """Test saving an ephemeral chat."""
    mock_app.current_chat_is_ephemeral = True
    mock_app.current_chat_conversation_id = None

    # Setup mock messages in the chat log
    mock_msg1 = MagicMock(spec=ChatMessage, role="User", message_text="Hello", generation_complete=True,
                          image_data=None, image_mime_type=None)
    mock_msg2 = MagicMock(spec=ChatMessage, role="AI", message_text="Hi", generation_complete=True, image_data=None,
                          image_mime_type=None)
    mock_app.query_one("#chat-log").query.return_value = [mock_msg1, mock_msg2]

    mock_ccl.create_conversation.return_value = "new_conv_id"

    await handle_chat_save_current_chat_button_pressed(mock_app, MagicMock())

    mock_ccl.create_conversation.assert_called_once()
    create_kwargs = mock_ccl.create_conversation.call_args.kwargs
    assert create_kwargs['title'].startswith("Chat: Hello...")
    assert len(create_kwargs['initial_messages']) == 2
    assert create_kwargs['initial_messages'][0]['content'] == "Hello"

    assert mock_app.current_chat_conversation_id == "new_conv_id"
    assert mock_app.current_chat_is_ephemeral is False
    mock_app.notify.assert_called_with("Chat saved successfully!", severity="information")
    mock_display_conv.assert_called_once_with(mock_app, "new_conv_id")


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ccl')
async def test_handle_chat_action_button_pressed_edit_and_save(mock_ccl, mock_app):
    """Test the edit->save workflow for a chat message."""
    mock_button = MagicMock(spec=Button, classes=["edit-button"])
    mock_action_widget = AsyncMock(spec=ChatMessage)
    mock_action_widget.message_text = "Original text"
    mock_action_widget.message_id_internal = "msg_123"
    mock_action_widget.message_version_internal = 0
    mock_action_widget._editing = False  # Start in non-editing mode
    mock_static_text = mock_action_widget.query_one.return_value

    # --- 1. First press: Start editing ---
    await handle_chat_action_button_pressed(mock_app, mock_button, mock_action_widget)

    mock_action_widget.mount.assert_called_once()  # Mounts the TextArea
    assert mock_action_widget._editing is True
    assert "💾" in mock_button.label  # Check for save emoji

    # --- 2. Second press: Save edit ---
    mock_action_widget._editing = True  # Simulate being in editing mode
    mock_edit_area = MagicMock(spec=TextArea, text="New edited text")
    mock_action_widget.query_one.return_value = mock_edit_area
    mock_ccl.edit_message_content.return_value = True

    await handle_chat_action_button_pressed(mock_app, mock_button, mock_action_widget)

    mock_edit_area.remove.assert_called_once()
    assert mock_action_widget.message_text == "New edited text"
    assert isinstance(mock_static_text.update.call_args[0][0], Text)
    assert mock_static_text.update.call_args[0][0].plain == "New edited text"

    mock_ccl.edit_message_content.assert_called_with(
        mock_app.chachanotes_db, "msg_123", "New edited text", 0
    )
    assert mock_action_widget.message_version_internal == 1  # Version incremented
    assert "✏️" in mock_button.label  # Check for edit emoji


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.load_character_and_image')
async def test_handle_chat_load_character_with_greeting(mock_load_char, mock_app):
    """Test that loading a character into an empty, ephemeral chat posts a greeting."""
    mock_app.current_chat_is_ephemeral = True
    mock_app.query_one("#chat-log").query.return_value = []  # Empty chat log

    char_data = {
        'id': 'char_abc', 'name': 'Greeter', 'first_message': 'Hello, adventurer!'
    }
    mock_load_char.return_value = (char_data, None, None)

    # Mock the list item from the character search list
    mock_list_item = MagicMock(spec=ListItem)
    mock_list_item.character_id = 'char_abc'
    mock_app.query_one("#chat-character-search-results-list").highlighted_child = mock_list_item

    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessage',
               new_callable=AsyncMock) as mock_chat_msg_class:
        await handle_chat_load_character_button_pressed(mock_app, MagicMock())

        # Assert character data is loaded
        assert mock_app.current_chat_active_character_data == char_data

        # Assert greeting message was created and mounted
        mock_chat_msg_class.assert_called_with(
            message='Hello, adventurer!',
            role='Greeter',
            generation_complete=True
        )
        mock_app.query_one("#chat-log").mount.assert_called_once_with(mock_chat_msg_class.return_value)