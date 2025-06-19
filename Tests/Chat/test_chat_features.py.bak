import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from textual.widgets import Button, TextArea, Static, Select, Checkbox, Input, Label
from textual.containers import VerticalScroll
from rich.text import Text

# Modules to be tested
from tldw_chatbook.Widgets.chat_message import ChatMessage
from tldw_chatbook.Event_Handlers.Chat_Events.chat_events import (
    handle_continue_response_button_pressed,
    handle_respond_for_me_button_pressed
)
# Mocked app class (simplified)
from tldw_chatbook.app import TldwCli
from tldw_chatbook.Character_Chat import Character_Chat_Lib as ccl


# Test Case 1: Thumbs Up/Down Icon Visibility
def test_thumbs_icons_visibility():
    """
    Tests the visibility of thumbs up/down icons in ChatMessage based on role.
    """
    # AI message should have thumbs up/down
    ai_message = ChatMessage(message="Hello AI", role="AI", generation_complete=True)
    ai_message_buttons = [widget.id for widget in ai_message.compose() if isinstance(widget, VerticalScroll)][0].query(
        Button)  # type: ignore

    ai_button_ids = {btn.id for btn in ai_message_buttons if btn.id}
    assert "thumb-up" in ai_button_ids
    assert "thumb-down" in ai_button_ids

    # User message should NOT have thumbs up/down
    # Role "User" will add class "-user" and not "-ai"
    user_message = ChatMessage(message="Hello User", role="User", generation_complete=True)

    # Need to properly compose to find buttons.
    # The compose() method yields ComposeResult, which is an iterable of widgets.
    # We need to find the Horizontal container for actions.
    user_actions_container = None
    for child in user_message.compose():  # Iterate through top-level children from compose
        if isinstance(child, VerticalScroll):  # The main container is Vertical
            # Find the Horizontal container with class 'message-actions'
            # This requires a bit more involved querying or knowledge of the structure.
            # For simplicity, let's assume we can query it after mounting (though we are not fully mounting here)
            # A more direct way is to check the logic within compose itself or inspect the structure.

            # Let's refine the check by finding the Horizontal actions container:
            actions_horizontal = None
            for sub_child in child.children:
                if isinstance(sub_child, Static) and sub_child.has_class("message-text"):  # skip header and text
                    continue
                if isinstance(sub_child, Label) and sub_child.has_class("message-header"):
                    continue
                if sub_child.has_class("message-actions"):  # This is the Horizontal container
                    actions_horizontal = sub_child
                    break

            user_message_buttons = []
            if actions_horizontal:
                user_message_buttons = actions_horizontal.query(Button)  # type: ignore
            break  # Found the main VerticalScroll

    user_button_ids = {btn.id for btn in user_message_buttons if btn.id}  # type: ignore
    assert "thumb-up" not in user_button_ids
    assert "thumb-down" not in user_button_ids


# Placeholder for async tests - requires pytest-asyncio
@pytest.mark.asyncio
async def test_placeholder_async():
    assert True


# More tests will be added below

# Mock TldwCli app for integration tests
@pytest_asyncio.fixture
async def mock_app_tldw(tmp_path):  # tmp_path is a pytest fixture for temporary directory
    # Create a minimal config file for the app to load
    # This avoids errors if the app tries to load a non-existent config
    default_config_content = """
[general]
log_level = "DEBUG"
default_tab = "chat"
USERS_NAME = "TestUser"

[api_settings.Ollama]
api_key_env_var = "OLLAMA_API_KEY" # Example, not actually used if chat_wrapper is mocked
streaming = true 

[chat_defaults]
provider = "Ollama"
model = "test_model"
system_prompt = "You are a helpful test assistant."
temperature = 0.7
top_p = 0.9
# Add other minimal required settings by the app or handlers
"""
    config_path = tmp_path / "test_config.toml"
    with open(config_path, "w") as f:
        f.write(default_config_content)

    # Patch DEFAULT_CONFIG_PATH before TldwCli is instantiated
    with patch('tldw_chatbook.config.DEFAULT_CONFIG_PATH', new=config_path):
        app = TldwCli()
        app.API_IMPORTS_SUCCESSFUL = True  # Assume API imports are fine for these tests

        # Mock essential UI components that are queried globally by handlers
        app.query_one = MagicMock()  # General mock for query_one

        # Mock specific queries needed by the handlers
        mock_chat_log = MagicMock(spec=VerticalScroll)
        mock_chat_log.query = MagicMock(return_value=[])  # Default to no messages

        mock_chat_input_area = MagicMock(spec=TextArea)
        mock_chat_input_area.text = ""

        mock_respond_button = MagicMock(spec=Button)
        mock_respond_button.label = "💡"

        # Configure query_one to return these mocks based on ID
        def side_effect_query_one(selector, *args, **kwargs):
            if selector == "#chat-log":
                return mock_chat_log
            elif selector == "#chat-input":
                return mock_chat_input_area
            elif selector == "#respond-for-me-button":
                return mock_respond_button
            # Mock parameters fetching for LLM calls
            elif selector == "#chat-api-provider":
                return MagicMock(spec=Select, value="Ollama")
            elif selector == "#chat-api-model":
                return MagicMock(spec=Select, value="test_model")
            elif selector == "#chat-system-prompt":
                return MagicMock(spec=TextArea, text="Test system prompt")
            elif selector == "#chat-temperature":
                return MagicMock(spec=Input, value="0.7")
            elif selector == "#chat-top-p":
                return MagicMock(spec=Input, value="0.9")
            elif selector == "#chat-min-p":
                return MagicMock(spec=Input, value="0.1")
            elif selector == "#chat-top-k":
                return MagicMock(spec=Input, value="40")
            elif selector == "#chat-llm-max-tokens":
                return MagicMock(spec=Input, value="512")
            elif selector == "#chat-llm-seed":
                return MagicMock(spec=Input, value="")  # No seed
            elif selector == "#chat-llm-stop":
                return MagicMock(spec=Input, value="")  # No stop sequences
            elif selector == "#chat-llm-response-format":
                return MagicMock(spec=Select, value="text")
            elif selector == "#chat-llm-n":
                return MagicMock(spec=Input, value="1")
            elif selector == "#chat-llm-user-identifier":
                return MagicMock(spec=Input, value="")
            elif selector == "#chat-llm-logprobs":
                return MagicMock(spec=Checkbox, value=False)
            elif selector == "#chat-llm-top-logprobs":
                return MagicMock(spec=Input, value="0")
            elif selector == "#chat-llm-logit-bias":
                return MagicMock(spec=TextArea, text="{}")
            elif selector == "#chat-llm-presence-penalty":
                return MagicMock(spec=Input, value="0.0")
            elif selector == "#chat-llm-frequency-penalty":
                return MagicMock(spec=Input, value="0.0")
            elif selector == "#chat-llm-tools":
                return MagicMock(spec=TextArea, text="[]")
            elif selector == "#chat-llm-tool-choice":
                return MagicMock(spec=Input, value="")
            elif selector == "#chat-llm-fixed-tokens-kobold":
                return MagicMock(spec=Checkbox, value=False)
            else:
                # Fallback for unhandled selectors: return a generic MagicMock
                # This helps avoid QueryError for components not directly involved in the test's core logic
                # but potentially touched by app setup or unrelated watchers.
                # Log a warning to indicate an unmocked query.
                print(f"Warning: Unmocked query_one selector: {selector}")
                return MagicMock()

        app.query_one = MagicMock(side_effect=side_effect_query_one)

        # Mock app.notify
        app.notify = MagicMock()

        # Mock app.chat_wrapper (will be an async generator for streaming, simple async for non-streaming)
        app.chat_wrapper = AsyncMock()  # General AsyncMock, can be configured per test

        # Mock DB
        app.chachanotes_db = MagicMock()
        if ccl:  # Ensure ccl is imported before trying to mock its methods
            ccl.edit_message_content = AsyncMock(return_value=True)  # Mock as async if it's called with await

        # Mock app_config (can be a simple dict for testing purposes)
        app.app_config = {
            "api_settings": {
                "ollama": {"streaming": True, "api_key_env_var": "OLLAMA_API_KEY"},  # Example
            },
            "general": {"log_level": "DEBUG", "default_tab": "chat", "USERS_NAME": "TestUser"},
            "chat_defaults": {"provider": "Ollama", "model": "test_model"}
        }
        # Mock get_char for emoji handling if it's directly used and matters for logic
        # For now, assume it returns fallback if not critical for test logic.

        # Simulate that the UI is ready to avoid issues with watchers
        app._ui_ready = True

        # Yield the app instance for use in tests
        yield app

        # Teardown (if any specific needed, though pytest fixtures handle tmp_path)
        app._ui_ready = False  # Reset after test


# Test Case 2: 'Continue Response' Button Functionality
@pytest.mark.asyncio
async def test_continue_response_button(mock_app_tldw: TldwCli):
    """Tests the 'Continue Response' button functionality."""
    app = mock_app_tldw

    # Setup: Create ChatMessage, Button, and mock chat log
    initial_text = "Original AI response."
    chat_message_widget = ChatMessage(
        message=initial_text,
        role="AI",
        generation_complete=True,
        message_id="test_msg_id_123",
        message_version=1
    )
    # Manually add the -ai class, as the constructor logic for class adding might not run in this isolated test
    chat_message_widget.add_class("-ai")

    # Mock the VerticalScroll (chat_log) to return our message widget
    # The mock_app_tldw.query_one for "#chat-log" already returns a MagicMock.
    # We need its .query(ChatMessage) method to return our widget.
    chat_log_mock = app.query_one("#chat-log")
    chat_log_mock.query = MagicMock(return_value=[chat_message_widget])

    # Mock the specific "continue-response-button" within the ChatMessage
    # This requires ChatMessage.compose() to have run, or mocking its query_one.
    # For simplicity, let's mock chat_message_widget.query_one directly for this button.
    mock_continue_button = MagicMock(spec=Button, id="continue-response-button")
    mock_continue_button.label = "↪️"  # Original label

    # Mock the .message-text Static widget within ChatMessage
    mock_static_text_widget = MagicMock(spec=Static)
    mock_static_text_widget.renderable = Text(initial_text)  # Store initial renderable Text
    mock_static_text_widget.update = MagicMock()  # Mock the update method

    def chat_message_query_one_side_effect(selector, *args, **kwargs):
        if selector == "#continue-response-button":
            return mock_continue_button
        elif selector == ".message-text":
            return mock_static_text_widget
        # Mock other buttons that might be disabled/enabled
        elif selector == "#thumb-up":
            return MagicMock(spec=Button)
        elif selector == "#thumb-down":
            return MagicMock(spec=Button)
        elif selector == "#regenerate":
            return MagicMock(spec=Button)
        raise AssertionError(f"chat_message_widget.query_one called with unmocked selector: {selector}")

    chat_message_widget.query_one = MagicMock(side_effect=chat_message_query_one_side_effect)

    # Configure app.chat_wrapper for streaming response
    stream_chunks = [" continued", " text", "."]

    async def mock_streaming_chat_wrapper(*args, **kwargs):
        for chunk in stream_chunks:
            yield chunk

    app.chat_wrapper.side_effect = mock_streaming_chat_wrapper

    # Mock ccl.edit_message_content if it's expected to be called
    if ccl:  # ccl might be None if import failed in some test environments
        ccl.edit_message_content = AsyncMock(return_value=True)  # Mock as non-async if it's not await'ed

    # Action
    await handle_continue_response_button_pressed(app, mock_continue_button, chat_message_widget)

    # Assertions
    # 1. Button state
    assert mock_continue_button.disabled is True  # Should be disabled during call
    # The handler re-enables it in the finally block
    # To test this, we need to ensure the label is restored too.
    # The check for `mock_continue_button.label == "↪️"` (original label) implicitly checks re-enablement if successful.
    # However, the test structure here runs the handler and then checks state.
    # The `finally` block *will* run. So it should be re-enabled.
    assert mock_continue_button.disabled is False
    assert mock_continue_button.label == "↪️"  # Restored label

    # 2. app.chat_wrapper call
    app.chat_wrapper.assert_called_once()
    call_args = app.chat_wrapper.call_args
    assert "Please continue generating the response" in call_args.kwargs['message']
    assert call_args.kwargs['history'][-1]['content'] == initial_text  # Last history item is the one to continue
    assert call_args.kwargs['streaming'] is True

    # 3. ChatMessage text update
    expected_final_text = initial_text + "".join(stream_chunks)
    assert chat_message_widget.message_text == expected_final_text
    # Check that static_text_widget.update was called to clear thinking indicator and then with final text
    # First call to update (once first chunk received) should be with original_text
    # Subsequent calls with appended chunks. The test setup for update mock is simple.
    # A more detailed check would involve call_args_list on mock_static_text_widget.update
    mock_static_text_widget.update.assert_any_call(Text(expected_final_text))

    # 4. DB call (ccl.edit_message_content)
    if ccl:  # Check if ccl and its mock are available
        ccl.edit_message_content.assert_called_once_with(
            app.chachanotes_db,
            "test_msg_id_123",
            expected_final_text,
            1  # original_message_version
        )
        assert chat_message_widget.message_version_internal == 2  # Version incremented

    # 5. Notification
    app.notify.assert_any_call("Message continuation saved to DB.", severity="information", timeout=2)


# Test Case 3: 'Respond for Me' Button Functionality
@pytest.mark.asyncio
async def test_respond_for_me_button(mock_app_tldw: TldwCli):
    """Tests the 'Respond for Me' button functionality."""
    app = mock_app_tldw

    # Setup: Mock chat history
    history_messages = [
        ChatMessage(message="User question?", role="User", generation_complete=True),
        ChatMessage(message="AI answer.", role="AI", generation_complete=True)
    ]
    chat_log_mock = app.query_one("#chat-log")
    chat_log_mock.query = MagicMock(return_value=history_messages)

    # Mock chat input TextArea
    chat_input_mock = app.query_one("#chat-input")  # Already mocked by fixture
    chat_input_mock.text = ""  # Ensure it's initially empty
    chat_input_mock.focus = MagicMock()

    # Mock the "Respond for Me" button itself (already mocked by fixture's query_one)
    respond_button_mock = app.query_one("#respond-for-me-button")
    original_button_label = respond_button_mock.label  # Store original label from mock setup

    # Configure app.chat_wrapper for non-streaming response
    suggested_response = "This is a suggested user response."
    app.chat_wrapper.side_effect = None  # Clear previous streaming side_effect
    app.chat_wrapper.return_value = suggested_response  # Direct return value for non-streaming

    # Action
    await handle_respond_for_me_button_pressed(app)

    # Assertions
    # 1. Button state
    # Initial disable and label change is internal to handler.
    # We check the final state after the `finally` block in the handler.
    assert respond_button_mock.disabled is False
    assert respond_button_mock.label == original_button_label  # Check if label is restored

    # 2. app.chat_wrapper call
    app.chat_wrapper.assert_called_once()
    call_args = app.chat_wrapper.call_args
    assert "Based on the following conversation, please suggest a concise and relevant response" in call_args.kwargs[
        'message']
    # History for "respond for me" is embedded in the prompt message itself, so history kwarg is empty
    assert call_args.kwargs['history'] == []
    assert call_args.kwargs['streaming'] is False

    # 3. Chat input update
    assert chat_input_mock.text == suggested_response.strip('"')  # Handler might strip quotes
    chat_input_mock.focus.assert_called_once()

    # 4. Notifications
    app.notify.assert_any_call("Generating suggestion...", timeout=2)
    app.notify.assert_any_call("Suggestion populated in the input field.", severity="information", timeout=3)
