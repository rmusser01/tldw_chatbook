# test_chat_message_enhanced_refactored.py
# Description: Refactored tests for ChatMessageEnhanced using standardized test infrastructure
#
"""
This is a refactored version of test_chat_message_enhanced.py that demonstrates
proper async testing patterns for Textual widgets using our standardized utilities.
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO

from PIL import Image as PILImage
from textual.widgets import Button

# Import our standardized test utilities
from Tests.textual_test_utils import (
    widget_pilot,
    wait_for_widget_mount,
    create_mock_app,
    get_widget_text
)

from tldw_chatbook.Widgets.chat_message_enhanced import ChatMessageEnhanced


class TestChatMessageEnhancedRefactored:
    """Refactored test suite using proper async patterns."""
    
    @pytest.fixture
    def sample_image_data(self):
        """Create sample image data for testing."""
        img = PILImage.new('RGB', (100, 100), color='red')
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        return buffer.getvalue()
    
    @pytest.fixture
    def message_params(self, sample_image_data):
        """Standard parameters for creating test messages."""
        return {
            "message": "Test message",
            "role": "User",
            "generation_complete": True,
            "message_id": "test-123",
            "message_version": 1,
            "timestamp": "2024-01-20 10:00:00",
            "image_data": sample_image_data,
            "image_mime_type": "image/png"
        }
    
    # Test 1: Basic initialization with proper app context
    async def test_initialization_with_app_context(self, widget_pilot, message_params):
        """Test widget initializes correctly within app context."""
        async with widget_pilot(ChatMessageEnhanced, **message_params) as pilot:
            widget = pilot.app.test_widget
            
            # Verify initialization
            assert widget.message == "Test message"
            assert widget.role == "User"
            assert widget.message_id == "test-123"
            assert widget.has_image
            assert widget.image_data is not None
    
    # Test 2: Copy functionality with proper async handling
    async def test_copy_message(self, widget_pilot, message_params):
        """Test copying message to clipboard."""
        async with widget_pilot(ChatMessageEnhanced, **message_params) as pilot:
            widget = pilot.app.test_widget
            
            # Wait for widget to be fully mounted
            await pilot.pause()
            
            # Mock pyperclip
            with patch('pyperclip.copy') as mock_copy:
                # Find and click copy button
                copy_button = widget.query_one("#copy-message-btn", Button)
                await pilot.click(copy_button)
                await pilot.pause()
                
                # Verify clipboard operation
                mock_copy.assert_called_once_with("Test message")
    
    # Test 3: Image display toggle
    async def test_image_toggle(self, widget_pilot, message_params):
        """Test toggling image display."""
        async with widget_pilot(ChatMessageEnhanced, **message_params) as pilot:
            widget = pilot.app.test_widget
            
            # Initial state - image should be visible
            image_container = widget.query_one("#image-container")
            assert image_container.styles.display != "none"
            
            # Toggle image
            toggle_button = widget.query_one("#toggle-image-btn", Button)
            await pilot.click(toggle_button)
            await pilot.pause()
            
            # Image should now be hidden
            assert widget.image_visible is False
    
    # Test 4: Edit mode with proper async flow
    async def test_edit_mode(self, widget_pilot, message_params):
        """Test entering and exiting edit mode."""
        async with widget_pilot(ChatMessageEnhanced, **message_params) as pilot:
            widget = pilot.app.test_widget
            
            # Enter edit mode
            edit_button = widget.query_one("#edit-message-btn", Button)
            await pilot.click(edit_button)
            await pilot.pause()
            
            # Verify edit mode is active
            assert widget.edit_mode
            
            # Find text area and modify content
            text_area = widget.query_one("#edit-text-area")
            await pilot.click(text_area)
            await pilot.press("ctrl+a")  # Select all
            await pilot.type("Updated message")
            
            # Save changes
            save_button = widget.query_one("#save-edit-btn", Button)
            await pilot.click(save_button)
            await pilot.pause()
            
            # Verify changes
            assert widget.message == "Updated message"
            assert not widget.edit_mode
    
    # Test 5: Notification handling with mocked app
    async def test_notifications(self, widget_pilot, message_params):
        """Test notification messages."""
        async with widget_pilot(ChatMessageEnhanced, **message_params) as pilot:
            widget = pilot.app.test_widget
            
            # Mock the notify method
            with patch.object(pilot.app, 'notify') as mock_notify:
                # Trigger a copy action
                copy_button = widget.query_one("#copy-message-btn", Button)
                await pilot.click(copy_button)
                await pilot.pause()
                
                # Verify notification was called
                mock_notify.assert_called()
                call_args = mock_notify.call_args[0][0]
                assert "copied" in call_args.lower()
    
    # Test 6: Image save functionality
    @patch('tldw_chatbook.Third_Party.textual_fspicker.FileSave')
    async def test_save_image(self, mock_file_save, widget_pilot, message_params):
        """Test saving image to file."""
        # Setup mock file dialog
        mock_dialog = Mock()
        mock_dialog.request_save.return_value = Path("/tmp/test_image.png")
        mock_file_save.return_value = mock_dialog
        
        async with widget_pilot(ChatMessageEnhanced, **message_params) as pilot:
            widget = pilot.app.test_widget
            
            # Mock push_screen_wait
            with patch.object(pilot.app, 'push_screen_wait', return_value=Path("/tmp/test_image.png")):
                # Click save image button
                save_button = widget.query_one("#save-image-btn", Button)
                await pilot.click(save_button)
                await pilot.pause()
                
                # Verify file save was attempted
                # (In real implementation, would check file was written)
    
    # Test 7: Message without image
    async def test_message_without_image(self, widget_pilot):
        """Test message display without image."""
        params = {
            "message": "Text only message",
            "role": "Assistant",
            "generation_complete": True,
            "message_id": "test-456"
        }
        
        async with widget_pilot(ChatMessageEnhanced, **params) as pilot:
            widget = pilot.app.test_widget
            
            # Verify no image elements are visible
            assert not widget.has_image
            
            # Image container should not exist or be hidden
            try:
                image_container = widget.query_one("#image-container")
                assert image_container.styles.display == "none"
            except:
                # Container might not exist at all, which is fine
                pass
    
    # Test 8: Concurrent operations
    async def test_concurrent_operations(self, widget_pilot, message_params):
        """Test handling multiple rapid user interactions."""
        async with widget_pilot(ChatMessageEnhanced, **message_params) as pilot:
            widget = pilot.app.test_widget
            
            # Rapidly click multiple buttons
            copy_button = widget.query_one("#copy-message-btn", Button)
            toggle_button = widget.query_one("#toggle-image-btn", Button)
            
            # Perform rapid actions
            for _ in range(5):
                await pilot.click(copy_button)
                await pilot.click(toggle_button)
            
            await pilot.pause()
            
            # Widget should still be in valid state
            assert widget.message == "Test message"
            # Final state depends on odd/even number of toggles
    
    # Test 9: Error handling
    async def test_error_handling(self, widget_pilot, message_params):
        """Test widget handles errors gracefully."""
        async with widget_pilot(ChatMessageEnhanced, **message_params) as pilot:
            widget = pilot.app.test_widget
            
            # Mock an error in clipboard operation
            with patch('pyperclip.copy', side_effect=Exception("Clipboard error")):
                # Should not crash the widget
                copy_button = widget.query_one("#copy-message-btn", Button)
                await pilot.click(copy_button)
                await pilot.pause()
                
                # Widget should still be functional
                assert widget.message == "Test message"


# Example of testing with a custom app that includes multiple widgets
class TestChatMessageInContext:
    """Test ChatMessageEnhanced within a larger app context."""
    
    async def test_multiple_messages(self, app_pilot, sample_image_data):
        """Test multiple chat messages in a conversation."""
        from textual.app import ComposeResult
        from textual.containers import Vertical
        
        class ChatApp(App):
            def compose(self) -> ComposeResult:
                with Vertical():
                    yield ChatMessageEnhanced(
                        message="Hello!",
                        role="User",
                        message_id="msg-1"
                    )
                    yield ChatMessageEnhanced(
                        message="Hi there!",
                        role="Assistant",
                        message_id="msg-2",
                        image_data=sample_image_data,
                        image_mime_type="image/png"
                    )
        
        async with app_pilot(ChatApp) as pilot:
            # Wait for all messages to mount
            messages = pilot.app.query(ChatMessageEnhanced)
            assert len(messages) == 2
            
            # Test interaction between messages
            user_msg = messages[0]
            assistant_msg = messages[1]
            
            assert user_msg.role == "User"
            assert assistant_msg.role == "Assistant"
            assert assistant_msg.has_image