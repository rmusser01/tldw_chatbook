# tests/Event_Handlers/Chat_Events/test_chat_events_integration.py
# Integration tests for chat event handlers using real Textual app

import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from textual.widgets import Button, Input, TextArea, Static, Select, ListView
from textual.containers import VerticalScroll

# Local imports
from tldw_chatbook.app import TldwCli
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase
from tldw_chatbook.Widgets.chat_message import ChatMessage
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl

# Event handlers to test
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import (
    handle_chat_send_button_pressed,
    handle_chat_action_button_pressed,
    handle_chat_new_conversation_button_pressed,
    handle_chat_save_current_chat_button_pressed,
    handle_chat_load_character_button_pressed,
    handle_chat_clear_active_character_button_pressed,
)

# Test marker for integration tests
pytestmark = [pytest.mark.integration, pytest.mark.skip(reason="Integration tests need refactoring for proper Textual app testing")]

#######################################################################################################################
#
# Fixtures

@pytest_asyncio.fixture
async def real_app(tmp_path):
    """Create a real TldwCli app instance for integration testing"""
    # Create temporary directories
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create config file
    config_content = f"""
[general]
log_level = "DEBUG"
default_tab = "chat"
USERS_NAME = "TestUser"

[paths]
data_dir = "{str(data_dir)}"
db_path = "{str(db_dir / 'chachanotes.db')}"
media_db_path = "{str(db_dir / 'media.db')}"

[chat_defaults]
provider = "OpenAI"
model = "gpt-3.5-turbo"
temperature = 0.7
max_tokens = 1000
streaming = true
system_prompt = "You are a helpful assistant."

[API]
OPENAI_API_KEY = "test-key-12345"
"""
    
    config_path = tmp_path / "config.toml"
    with open(config_path, "w") as f:
        f.write(config_content)
    
    # Set environment variable
    os.environ['TLDW_CONFIG_PATH'] = str(config_path)
    
    # Create app instance
    app = TldwCli()
    app.API_IMPORTS_SUCCESSFUL = True
    
    # Initialize databases
    app.chachanotes_db = CharactersRAGDB(str(db_dir / "chachanotes.db"), "test_user")
    app.media_db = MediaDatabase(str(db_dir / "media.db"), client_id="test_client_id")
    
    # Set app attributes
    app.current_chat_conversation_id = None
    app.current_chat_is_ephemeral = True
    app.current_chat_active_character_data = None
    app.notes_user_id = "test_user"
    
    # Add missing attributes and methods that the real app would have
    app.app_config = {
        'chat_defaults': {
            'provider': 'OpenAI',
            'model': 'gpt-3.5-turbo',
            'temperature': 0.7,
            'max_tokens': 1000,
            'streaming': True,
            'system_prompt': 'You are a helpful assistant.',
            'strip_thinking_tags': True
        },
        'api_settings': {
            'openai': {
                'streaming': True,
                'api_key': 'test-key-12345'
            }
        }
    }
    
    # Mock the chat_wrapper method
    app.chat_wrapper = MagicMock(return_value="Test response")
    
    # Mock the notify method (Textual apps have this)
    app.notify = MagicMock()
    
    # Add missing methods
    app.set_current_chat_worker = MagicMock()
    app.set_current_ai_message_widget = MagicMock()
    app.get_current_ai_message_widget = MagicMock(return_value=None)
    
    # Add missing attributes
    app.current_ai_message_widget = None
    app.current_chat_is_streaming = False
    app.current_chat_worker = None
    app._chat_state_lock = MagicMock()
    
    yield app
    
    # Cleanup - databases don't have close() method


# Helper functions

async def setup_chat_ui(app):
    """Helper to set up chat UI elements for testing"""
    async with app.run_test() as pilot:
        await pilot.pause(0.5)
        # Navigate to chat tab
        await pilot.press("c")
        await pilot.pause(0.1)
        return pilot


def create_test_character(db: CharactersRAGDB, name: str = "Test Character") -> dict:
    """Helper to create a test character"""
    char_id = ccl.add_character(
        db,
        name=name,
        description="A test character",
        personality="Helpful and friendly",
        scenario="Test scenario",
        system_prompt="You are a test character.",
        post_history_instructions="",
        first_mes="Hello, I'm a test character!",
        mes_example="",
        creator_notes="",
        image="",
        creator="test",
        character_version="1.0",
        extensions={},
        avatar_path=None
    )
    return ccl.get_character_by_id(db, char_id)


def create_test_conversation(db: CharactersRAGDB, title: str = "Test Chat") -> str:
    """Helper to create a test conversation"""
    conv_id = ccl.create_conversation(
        db,
        title=title,
        character_id=None,
        initial_messages=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]
    )
    return conv_id


#######################################################################################################################
#
# Test Classes

class TestChatBasicOperations:
    """Test basic chat operations with real app"""
    
    @pytest.mark.asyncio
    async def test_send_message(self, real_app):
        """Test sending a message in chat"""
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Get chat input
            chat_input = real_app.query_one("#chat-input", TextArea)
            assert chat_input is not None
            
            # Type a message
            await pilot.click(chat_input)
            # Set the text directly on the TextArea widget
            chat_input.text = "Hello, this is a test message"
            
            # Mock the chat API call to avoid real API requests
            with patch.object(real_app, 'chat_wrapper', return_value="Test response"):
                # Click send button
                send_button = real_app.query_one("#chat-send-button", Button)
                await pilot.click(send_button)
                await pilot.pause(0.2)
            
            # Check that input was cleared
            assert chat_input.text == ""
            
            # Check that messages were added to chat log
            chat_log = real_app.query_one("#chat-log", VerticalScroll)
            messages = chat_log.query(ChatMessage)
            assert len(messages) >= 1  # At least user message
            
            # Verify user message
            user_msg = messages[0]
            assert user_msg.role == "User"
            assert user_msg.message_text == "Hello, this is a test message"
    
    @pytest.mark.asyncio
    async def test_new_conversation(self, real_app):
        """Test creating a new conversation"""
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Set some initial state
            real_app.current_chat_conversation_id = "old_conv_id"
            real_app.current_chat_is_ephemeral = False
            
            # Add some messages to chat log
            chat_log = real_app.query_one("#chat-log", VerticalScroll)
            msg1 = ChatMessage(message="Old message", role="User")
            await chat_log.mount(msg1)
            
            # Click new conversation button
            new_button = real_app.query_one("#chat-new-conversation-button", Button)
            await pilot.click(new_button)
            await pilot.pause(0.1)
            
            # Check state was reset
            assert real_app.current_chat_conversation_id is None
            assert real_app.current_chat_is_ephemeral is True
            
            # Check chat log was cleared
            messages = chat_log.query(ChatMessage)
            assert len(messages) == 0
    
    @pytest.mark.asyncio
    async def test_save_ephemeral_chat(self, real_app):
        """Test saving an ephemeral chat"""
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Ensure we're in ephemeral mode
            real_app.current_chat_is_ephemeral = True
            real_app.current_chat_conversation_id = None
            
            # Add some messages to chat log
            chat_log = real_app.query_one("#chat-log", VerticalScroll)
            msg1 = ChatMessage(message="Test message 1", role="User", generation_complete=True)
            msg2 = ChatMessage(message="Test response", role="AI", generation_complete=True)
            await chat_log.mount(msg1, msg2)
            await pilot.pause(0.1)
            
            # Click save button
            save_button = real_app.query_one("#chat-save-current-chat-button", Button)
            await pilot.click(save_button)
            await pilot.pause(0.2)
            
            # Check that chat was saved
            assert real_app.current_chat_is_ephemeral is False
            assert real_app.current_chat_conversation_id is not None
            
            # Verify conversation exists in database
            conv = ccl.get_conversation_by_id(
                real_app.chachanotes_db,
                real_app.current_chat_conversation_id
            )
            assert conv is not None
            assert conv['title'].startswith("Chat:")


class TestChatCharacterIntegration:
    """Test character-related chat functionality"""
    
    @pytest.mark.asyncio
    async def test_load_character(self, real_app):
        """Test loading a character into chat"""
        # Create a test character
        character = create_test_character(real_app.chachanotes_db, "Assistant Bot")
        
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Mock the character selection
            with patch('tldw_chatbook.Event_Handlers.Chat_Events.chat_events.display_available_characters_for_chat',
                      return_value=character['id']):
                # Click load character button
                load_char_button = real_app.query_one("#chat-load-character-button", Button)
                await pilot.click(load_char_button)
                await pilot.pause(0.2)
            
            # Check character was loaded
            assert real_app.current_chat_active_character_data is not None
            assert real_app.current_chat_active_character_data['name'] == "Assistant Bot"
            
            # Check system prompt was updated
            system_prompt_area = real_app.query_one("#chat-system-prompt", TextArea)
            assert character['system_prompt'] in system_prompt_area.text
    
    @pytest.mark.asyncio
    async def test_clear_active_character(self, real_app):
        """Test clearing the active character"""
        # Set up an active character
        character = create_test_character(real_app.chachanotes_db)
        real_app.current_chat_active_character_data = character
        
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Click clear character button
            clear_button = real_app.query_one("#chat-clear-active-character-button", Button)
            await pilot.click(clear_button)
            await pilot.pause(0.1)
            
            # Check character was cleared
            assert real_app.current_chat_active_character_data is None
            
            # Check system prompt was reset to default
            system_prompt_area = real_app.query_one("#chat-system-prompt", TextArea)
            default_prompt = real_app.app_config['chat_defaults']['system_prompt']
            assert default_prompt in system_prompt_area.text


class TestChatMessageEditing:
    """Test message editing functionality"""
    
    @pytest.mark.asyncio
    async def test_edit_message(self, real_app):
        """Test editing a chat message"""
        # Create a conversation with messages
        conv_id = create_test_conversation(real_app.chachanotes_db)
        real_app.current_chat_conversation_id = conv_id
        real_app.current_chat_is_ephemeral = False
        
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Add a message to edit
            chat_log = real_app.query_one("#chat-log", VerticalScroll)
            msg = ChatMessage(
                message="Original message text",
                role="User",
                message_id="msg_123",
                message_version=1,
                generation_complete=True
            )
            await chat_log.mount(msg)
            await pilot.pause(0.1)
            
            # Find edit button
            edit_buttons = msg.query(".edit-button")
            assert len(edit_buttons) > 0
            edit_button = edit_buttons[0]
            
            # Click edit button to start editing
            await pilot.click(edit_button)
            await pilot.pause(0.1)
            
            # Check that edit area was created
            edit_area = msg.query_one("#edit-area", TextArea)
            assert edit_area is not None
            assert edit_area.text == "Original message text"
            
            # Edit the text
            await pilot.click(edit_area)
            await pilot.press("ctrl+a")  # Select all
            await pilot.type("Edited message text")
            
            # Click save button (same button, now shows save)
            await pilot.click(edit_button)
            await pilot.pause(0.1)
            
            # Check message was updated
            assert msg.message_text == "Edited message text"
            
            # Verify in database
            messages = ccl.get_messages_for_conversation(real_app.chachanotes_db, conv_id)
            edited_msg = next((m for m in messages if m['id'] == "msg_123"), None)
            assert edited_msg is not None
            assert edited_msg['content'] == "Edited message text"
            assert edited_msg['version'] == 2  # Version incremented


class TestChatWithStreaming:
    """Test chat with streaming responses"""
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, real_app):
        """Test handling streaming responses"""
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Mock streaming response
            async def mock_streaming_chat(*args, **kwargs):
                # Simulate streaming by yielding chunks
                for chunk in ["Hello", " ", "from", " ", "streaming", "!"]:
                    yield chunk
            
            with patch.object(real_app, 'chat_wrapper', side_effect=mock_streaming_chat):
                # Send a message
                chat_input = real_app.query_one("#chat-input", TextArea)
                await pilot.click(chat_input)
                await pilot.type("Test streaming")
                
                send_button = real_app.query_one("#chat-send-button", Button)
                await pilot.click(send_button)
                await pilot.pause(0.5)
                
                # Check that streaming message was created
                chat_log = real_app.query_one("#chat-log", VerticalScroll)
                messages = chat_log.query(ChatMessage)
                
                # Find AI message
                ai_messages = [m for m in messages if m.role == "AI"]
                assert len(ai_messages) > 0
                
                # Final message should be complete
                ai_msg = ai_messages[-1]
                assert ai_msg.generation_complete is True
                assert ai_msg.message_text == "Hello from streaming!"


class TestChatErrorHandling:
    """Test error handling in chat operations"""
    
    @pytest.mark.asyncio
    async def test_api_error_handling(self, real_app):
        """Test handling of API errors"""
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Mock API error
            with patch.object(real_app, 'chat_wrapper', side_effect=Exception("API Error")):
                # Send a message
                chat_input = real_app.query_one("#chat-input", TextArea)
                await pilot.click(chat_input)
                await pilot.type("Test error")
                
                send_button = real_app.query_one("#chat-send-button", Button)
                await pilot.click(send_button)
                await pilot.pause(0.2)
                
                # Check that error was handled gracefully
                # App should not crash and should show notification
                assert real_app.notify.called
                notify_args = real_app.notify.call_args[0]
                assert "error" in notify_args[0].lower()
    
    @pytest.mark.asyncio
    async def test_database_error_handling(self, real_app):
        """Test handling of database errors"""
        async with real_app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Mock database error
            with patch.object(ccl, 'create_conversation', side_effect=Exception("DB Error")):
                # Try to save ephemeral chat
                real_app.current_chat_is_ephemeral = True
                
                # Add a message
                chat_log = real_app.query_one("#chat-log", VerticalScroll)
                msg = ChatMessage(message="Test", role="User", generation_complete=True)
                await chat_log.mount(msg)
                
                save_button = real_app.query_one("#chat-save-current-chat-button", Button)
                await pilot.click(save_button)
                await pilot.pause(0.2)
                
                # Check error was handled
                assert real_app.notify.called
                notify_args = real_app.notify.call_args[0]
                assert "error" in notify_args[0].lower()
                
                # Chat should still be ephemeral
                assert real_app.current_chat_is_ephemeral is True