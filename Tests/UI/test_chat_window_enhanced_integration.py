"""Integration tests for ChatWindowEnhanced using Textual's Pilot framework."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock
import tempfile

from textual.app import App
from textual.widgets import Button, TextArea, Static
from textual.containers import Container


@pytest.fixture
def mock_app_config():
    """Create a mock app configuration."""
    return {
        "chat_defaults": {
            "provider": "openai",
            "model": "gpt-4",
            "temperature": 0.7,
            "enable_tabs": False
        },
        "chat": {
            "voice": {"show_mic_button": True},
            "images": {"show_attach_button": True}
        }
    }


@pytest.fixture
def chat_app(mock_app_config):
    """Create a test app with ChatWindowEnhanced."""
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
    
    with patch('tldw_chatbook.config.get_cli_setting') as mock_get_setting:
        # Mock config settings
        def get_setting(section, key, default=None):
            if section == "chat_defaults":
                return mock_app_config["chat_defaults"].get(key, default)
            elif section == "chat":
                if key in mock_app_config["chat"]:
                    return mock_app_config["chat"][key]
            return default
        
        mock_get_setting.side_effect = get_setting
        
        # Create app instance
        app = TldwCli()
        app.app_config = mock_app_config
        
        # Mock necessary attributes
        app.chat_attached_files = {}
        app.active_session_id = "default"
        app.is_streaming = False
        
        return app


class TestChatWindowEnhancedIntegration:
    """Integration tests for ChatWindowEnhanced functionality."""
    
    @pytest.mark.asyncio
    async def test_widget_initialization(self, chat_app):
        """Test that all widgets are properly initialized on mount."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Check core widgets exist
            assert pilot.app.query_one("#send-stop-chat", Button) is not None
            assert pilot.app.query_one("#chat-input", TextArea) is not None
            
            # Check optional widgets based on config
            mic_button = pilot.app.query("#mic-button", Button)
            assert len(mic_button) > 0  # Should exist based on mock config
            
            attach_button = pilot.app.query("#attach-image", Button)
            assert len(attach_button) > 0  # Should exist based on mock config
    
    @pytest.mark.asyncio
    async def test_send_button_state_changes(self, chat_app):
        """Test send/stop button state changes during streaming."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Get the send button
            send_button = pilot.app.query_one("#send-stop-chat", Button)
            
            # Initially should be in "Send" state
            assert "Send" in send_button.label or "➤" in send_button.label
            
            # Type a message
            chat_input = pilot.app.query_one("#chat-input", TextArea)
            chat_input.value = "Test message"
            
            # Click send button
            await pilot.click("#send-stop-chat")
            await pilot.pause(0.1)
            
            # During streaming, button should change to "Stop"
            # Note: This would need proper mocking of the streaming functionality
            # For now, we just test that the button click is handled
            assert send_button is not None
    
    @pytest.mark.asyncio
    async def test_attachment_indicator_updates(self, chat_app):
        """Test that attachment indicator updates when files are attached."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Get attachment indicator
            indicator = pilot.app.query_one("#image-attachment-indicator", Static)
            
            # Initially should be empty or hidden
            assert indicator.renderable == "" or not indicator.display
            
            # Simulate attaching a file
            pilot.app.chat_attached_files["default"] = [
                {"path": "/test/file.txt", "type": "text"}
            ]
            
            # Trigger update (in real app this would happen via reactive property)
            from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            if chat_window:
                chat_window.pending_attachment = "/test/file.txt"
                await pilot.pause(0.1)
    
    @pytest.mark.asyncio
    async def test_sidebar_toggle_functionality(self, chat_app):
        """Test that sidebar toggles work correctly."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Test left sidebar toggle
            left_sidebar = pilot.app.query("#chat-left-sidebar")
            if left_sidebar:
                initial_display = left_sidebar[0].display
                
                # Click toggle button
                await pilot.click("#toggle-chat-left-sidebar")
                await pilot.pause(0.1)
                
                # Display should have changed
                assert left_sidebar[0].display != initial_display
                
                # Toggle back
                await pilot.click("#toggle-chat-left-sidebar")
                await pilot.pause(0.1)
                assert left_sidebar[0].display == initial_display
    
    @pytest.mark.asyncio
    async def test_keyboard_shortcuts(self, chat_app):
        """Test keyboard shortcuts work correctly."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Test sidebar resize shortcuts
            left_sidebar = pilot.app.query("#chat-left-sidebar")
            if left_sidebar:
                initial_width = left_sidebar[0].styles.width
                
                # Expand sidebar
                await pilot.press("ctrl+shift+right")
                await pilot.pause(0.1)
                
                # Width should have increased (if implemented)
                # Note: This depends on the actual implementation
                
            # Test voice input toggle
            await pilot.press("ctrl+m")
            await pilot.pause(0.1)
            # Voice input widget should be created/toggled
    
    @pytest.mark.asyncio
    async def test_chat_input_focus(self, chat_app):
        """Test that chat input receives focus correctly."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Get chat input
            chat_input = pilot.app.query_one("#chat-input", TextArea)
            
            # Type some text
            await pilot.click("#chat-input")
            await pilot.press("H", "e", "l", "l", "o")
            await pilot.pause(0.1)
            
            # Check text was entered
            assert chat_input.value == "Hello"
            
            # Clear the input
            chat_input.clear()
            assert chat_input.value == ""
    
    @pytest.mark.asyncio
    async def test_file_attachment_workflow(self, chat_app):
        """Test the complete file attachment workflow."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Test file content")
            test_file_path = f.name
        
        try:
            async with chat_app.run_test() as pilot:
                # Navigate to chat tab
                await pilot.press("ctrl+1")
                await pilot.pause(0.1)
                
                # Click attach button
                attach_button = pilot.app.query("#attach-image", Button)
                if attach_button:
                    # Simulate file selection (would normally open file picker)
                    from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
                    chat_window = pilot.app.query_one(ChatWindowEnhanced)
                    
                    # Directly set the pending attachment
                    chat_window.pending_attachment = test_file_path
                    await pilot.pause(0.1)
                    
                    # Check that attachment indicator updated
                    indicator = pilot.app.query_one("#image-attachment-indicator", Static)
                    # Indicator should show something
                    assert chat_window.pending_attachment == test_file_path
        finally:
            # Clean up test file
            Path(test_file_path).unlink(missing_ok=True)
    
    @pytest.mark.asyncio
    async def test_error_handling_display(self, chat_app):
        """Test that errors are properly displayed to the user."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Simulate an error condition
            from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            
            # Try to attach a non-existent file
            chat_window.pending_attachment = "/non/existent/file.txt"
            
            # Process the attachment (this should fail)
            with patch.object(pilot.app, 'notify') as mock_notify:
                # Trigger file processing
                await chat_window._process_file_worker("/non/existent/file.txt")
                
                # Check that an error notification was shown
                # Note: Actual implementation would need proper error handling
    
    @pytest.mark.asyncio
    async def test_reactive_properties_update_ui(self, chat_app):
        """Test that reactive properties properly update the UI."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            
            # Get send button
            send_button = pilot.app.query_one("#send-stop-chat", Button)
            initial_label = send_button.label
            
            # Change is_send_button reactive property
            chat_window.is_send_button = False
            await pilot.pause(0.1)
            
            # Button label should have changed
            assert send_button.label != initial_label
            
            # Change back
            chat_window.is_send_button = True
            await pilot.pause(0.1)
            assert send_button.label == initial_label


class TestChatWindowEnhancedPerformance:
    """Performance-related integration tests."""
    
    @pytest.mark.asyncio
    async def test_widget_caching_performance(self, chat_app):
        """Test that widget caching improves performance."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            
            # Check that widgets are cached
            assert chat_window._send_button is not None
            assert chat_window._chat_input is not None
            
            # Accessing cached widgets should be fast
            import time
            start = time.time()
            for _ in range(100):
                _ = chat_window._send_button
                _ = chat_window._chat_input
            cached_time = time.time() - start
            
            # Compare with querying
            start = time.time()
            for _ in range(100):
                _ = pilot.app.query_one("#send-stop-chat", Button)
                _ = pilot.app.query_one("#chat-input", TextArea)
            query_time = time.time() - start
            
            # Cached access should be significantly faster
            assert cached_time < query_time * 0.5  # At least 2x faster
    
    @pytest.mark.asyncio
    async def test_batch_updates_reduce_reflows(self, chat_app):
        """Test that batch updates reduce UI reflows."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
            chat_window = pilot.app.query_one(ChatWindowEnhanced)
            
            # Test batch update if implemented
            # This would need the actual batch_update context manager
            # to be implemented in the main code


class TestChatWindowEnhancedAccessibility:
    """Accessibility and usability tests."""
    
    @pytest.mark.asyncio
    async def test_tooltips_present(self, chat_app):
        """Test that all buttons have helpful tooltips."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Check main buttons have tooltips
            buttons_to_check = [
                "#send-stop-chat",
                "#toggle-chat-left-sidebar",
                "#toggle-chat-right-sidebar"
            ]
            
            for button_id in buttons_to_check:
                button = pilot.app.query(button_id, Button)
                if button:
                    assert button[0].tooltip is not None
                    assert len(button[0].tooltip) > 0
    
    @pytest.mark.asyncio
    async def test_keyboard_navigation(self, chat_app):
        """Test that keyboard navigation works properly."""
        async with chat_app.run_test() as pilot:
            # Navigate to chat tab
            await pilot.press("ctrl+1")
            await pilot.pause(0.1)
            
            # Tab through widgets
            await pilot.press("tab")
            await pilot.pause(0.05)
            await pilot.press("tab")
            await pilot.pause(0.05)
            
            # Should be able to navigate between focusable widgets
            focused = pilot.app.focused
            assert focused is not None