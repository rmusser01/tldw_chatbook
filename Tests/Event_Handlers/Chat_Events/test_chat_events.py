# /tests/Event_Handlers/Chat_Events/test_chat_events.py
# Unit tests for chat event handlers using mocked components

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
from tldw_chatbook.Utils.Emoji_Handling import (
    EMOJI_SAVE_EDIT, FALLBACK_SAVE_EDIT, EMOJI_EDIT, FALLBACK_EDIT
)
from tldw_chatbook.Widgets.chat_message import ChatMessage
from tldw_chatbook.Widgets.chat_message_enhanced import ChatMessageEnhanced

# Import our comprehensive mock fixture
from Tests.fixtures.event_handler_mocks import mock_app

# Test marker for unit tests
pytestmark = [pytest.mark.asyncio, pytest.mark.unit]


# The mock_app fixture is imported from Tests.fixtures.event_handler_mocks


# Mock external dependencies used in chat_events.py
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ccl')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.os')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessageEnhanced')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessage')
async def test_handle_chat_send_button_pressed_basic(mock_chat_message_class, mock_chat_message_enhanced_class, mock_os, mock_ccl, mock_app):
    """Test a basic message send operation."""
    mock_os.environ.get.return_value = "fake-key"
    
    # Mock ChatMessage instances to track mount calls
    mock_user_msg = MagicMock()
    mock_ai_msg = MagicMock()
    # Could be either ChatMessage or ChatMessageEnhanced depending on config
    mock_chat_message_class.side_effect = [mock_user_msg, mock_ai_msg]
    mock_chat_message_enhanced_class.side_effect = [mock_user_msg, mock_ai_msg]

    await handle_chat_send_button_pressed(mock_app, MagicMock())

    # Assert UI updates
    # TextArea.clear is sync, not async
    mock_app.query_one("#chat-input").clear.assert_called_once()
    # Check that mount was called for user message and AI placeholder
    # Get all mount calls
    mount_calls = mock_app.query_one("#chat-log").mount.call_args_list
    # Should have exactly 2 mounts: user message and AI placeholder
    assert len(mount_calls) == 2
    # Verify the mounted objects are the mocked ChatMessage instances
    assert mount_calls[0][0][0] == mock_user_msg  # First mount is user message
    assert mount_calls[1][0][0] == mock_ai_msg    # Second mount is AI placeholder

    # Assert worker is called
    mock_app.run_worker.assert_called_once()

    # Assert chat_wrapper is called with correct parameters by the worker
    worker_lambda = mock_app.run_worker.call_args[0][0]
    worker_lambda()  # Execute the lambda to trigger the call to chat_wrapper

    # chat_wrapper is AsyncMock, so we need to check it was called (not await it in test)
    mock_app.chat_wrapper.assert_called_once()
    wrapper_kwargs = mock_app.chat_wrapper.call_args.kwargs
    assert wrapper_kwargs['message'] == "User message"
    assert wrapper_kwargs['api_endpoint'] == "OpenAI"
    assert wrapper_kwargs['api_key'] == "fake-key"
    assert wrapper_kwargs['system_message'] == "UI system prompt"
    assert wrapper_kwargs['streaming'] is True  # From config


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ccl')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.os')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessageEnhanced')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessage')
async def test_handle_chat_send_with_active_character(mock_chat_message_class, mock_chat_message_enhanced_class, mock_os, mock_ccl, mock_app):
    """Test that an active character's system prompt overrides the UI."""
    mock_os.environ.get.return_value = "fake-key"
    
    # Mock ChatMessage instances
    mock_user_msg = MagicMock()
    mock_ai_msg = MagicMock()
    mock_chat_message_class.side_effect = [mock_user_msg, mock_ai_msg]
    mock_chat_message_enhanced_class.side_effect = [mock_user_msg, mock_ai_msg]
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


@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.TextArea')
@patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ccl')
async def test_handle_chat_action_button_pressed_edit_and_save(mock_ccl, mock_textarea_class, mock_app):
    """Test the edit->save workflow for a chat message."""
    mock_button = MagicMock(spec=Button, classes=["edit-button"])
    
    # Create a proper mock action widget using MagicMock, not AsyncMock
    mock_action_widget = MagicMock(spec=ChatMessage)
    mock_action_widget.message_text = "Original text"
    mock_action_widget.message_id_internal = "msg_123"
    mock_action_widget.message_version_internal = 0
    mock_action_widget._editing = False  # Start in non-editing mode
    mock_action_widget.mount = AsyncMock()  # mount is async
    mock_action_widget.remove = AsyncMock()  # remove is async
    
    # Create mock for the static text widget
    mock_static_text = MagicMock(spec=Static)
    mock_static_text.update = MagicMock()  # update is sync
    mock_action_widget.query_one = MagicMock(return_value=mock_static_text)

    # --- 1. First press: Start editing ---
    # Mock the TextArea creation
    mock_editor = MagicMock(spec=TextArea)
    mock_editor.styles = MagicMock()
    mock_editor.focus = MagicMock()  # focus is sync
    mock_textarea_class.return_value = mock_editor
    
    await handle_chat_action_button_pressed(mock_app, mock_button, mock_action_widget)

    mock_action_widget.mount.assert_called_once_with(mock_editor, before=mock_static_text)
    assert mock_action_widget._editing is True
    # Check for save emoji or fallback text
    assert EMOJI_SAVE_EDIT in mock_button.label or FALLBACK_SAVE_EDIT in mock_button.label

    # --- 2. Second press: Save edit ---
    mock_action_widget._editing = True  # Simulate being in editing mode
    # Change what query_one returns for the edit area
    mock_editor.text = "New edited text"
    mock_editor.remove = AsyncMock()  # remove is async
    
    def query_one_side_effect(selector, widget_type=None):
        if selector == "#edit-area":
            return mock_editor
        return mock_static_text
    
    mock_action_widget.query_one.side_effect = query_one_side_effect
    mock_ccl.edit_message_content.return_value = True

    await handle_chat_action_button_pressed(mock_app, mock_button, mock_action_widget)

    mock_editor.remove.assert_called_once()
    assert mock_action_widget.message_text == "New edited text"
    assert isinstance(mock_static_text.update.call_args[0][0], Text)
    assert mock_static_text.update.call_args[0][0].plain == "New edited text"

    mock_ccl.edit_message_content.assert_called_with(
        mock_app.chachanotes_db, "msg_123", "New edited text", 0
    )
    assert mock_action_widget.message_version_internal == 1  # Version incremented
    # Check for edit emoji or fallback text
    assert EMOJI_EDIT in mock_button.label or FALLBACK_EDIT in mock_button.label


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

    # Use MagicMock instead of AsyncMock for ChatMessage since it's not async
    with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.ChatMessage') as mock_chat_msg_class:
        # Create a proper mock instance
        mock_greeting_msg = MagicMock(spec=ChatMessage)
        mock_chat_msg_class.return_value = mock_greeting_msg
        await handle_chat_load_character_button_pressed(mock_app, MagicMock())

        # Assert character data is loaded
        assert mock_app.current_chat_active_character_data == char_data

        # Assert greeting message was created and mounted
        mock_chat_msg_class.assert_called_with(
            message='Hello, adventurer!',
            role='Greeter',
            generation_complete=True
        )
        mock_app.query_one("#chat-log").mount.assert_called_once_with(mock_greeting_msg)