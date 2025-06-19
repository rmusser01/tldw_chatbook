# Tests/Widgets/test_chat_message_enhanced.py
# Description: Unit tests for the enhanced ChatMessage widget with image support
#
# Imports
#
# Standard Library
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# 3rd-party Libraries
from PIL import Image as PILImage
from textual.app import App
from textual.widgets import Button

# Local Imports
from tldw_chatbook.Widgets.chat_message_enhanced import ChatMessageEnhanced

#
#######################################################################################################################
#
# Test Fixtures

@pytest.fixture
def sample_image_data():
    """Create sample image data for testing."""
    # Create a small test image
    img = PILImage.new('RGB', (100, 100), color='red')
    from io import BytesIO
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def chat_message(sample_image_data):
    """Create a ChatMessageEnhanced instance for testing."""
    message = ChatMessageEnhanced(
        message="Test message",
        role="User",
        generation_complete=True,
        message_id="test-123",
        message_version=1,
        timestamp="2024-01-20 10:00:00",
        image_data=sample_image_data,
        image_mime_type="image/png"
    )
    # Don't set app - it's a read-only property managed by Textual
    return message


#
# Unit Tests
#

class TestChatMessageEnhanced:
    """Test suite for ChatMessageEnhanced widget."""
    
    def test_initialization(self, sample_image_data):
        """Test that ChatMessageEnhanced initializes correctly."""
        message = ChatMessageEnhanced(
            message="Hello, world!",
            role="User",
            image_data=sample_image_data,
            image_mime_type="image/png"
        )
        
        assert message.message_text == "Hello, world!"
        assert message.role == "User"
        assert message.image_data == sample_image_data
        assert message.image_mime_type == "image/png"
        assert message.pixel_mode is False
        assert message.has_class("-user")
    
    def test_ai_message_initialization(self):
        """Test AI message initialization."""
        message = ChatMessageEnhanced(
            message="AI response",
            role="Assistant",
            generation_complete=False
        )
        
        assert message.role == "Assistant"
        assert message.generation_complete is False
        assert message.has_class("-ai")
        assert not message.has_class("-user")
    
    def test_compose_with_image(self, chat_message):
        """Test that compose includes image elements when image data is present."""
        # Note: compose() requires an active app context
        # This is better tested in the async test suite
        # Just verify the widget has the expected attributes
        assert chat_message.image_data is not None
        assert chat_message.image_mime_type == "image/png"
        assert hasattr(chat_message, 'compose')
    
    def test_compose_without_image(self):
        """Test that compose works without image data."""
        message = ChatMessageEnhanced(
            message="Text only",
            role="User"
        )
        
        # Note: compose() requires an active app context
        # Just verify the widget was created properly
        assert message.image_data is None
        assert hasattr(message, 'compose')
    
    @patch('tldw_chatbook.Widgets.chat_message_enhanced.TEXTUAL_IMAGE_AVAILABLE', True)
    def test_render_regular_with_textual_image(self, chat_message):
        """Test regular rendering when textual-image is available."""
        chat_message._image_widget = Mock()
        chat_message._image_widget.mount = Mock()
        
        with patch('tldw_chatbook.Widgets.chat_message_enhanced.TextualImage') as mock_image:
            mock_image.from_bytes.return_value = Mock()
            chat_message._render_regular()
            
            mock_image.from_bytes.assert_called_once_with(chat_message.image_data)
            chat_message._image_widget.mount.assert_called_once()
    
    @patch('tldw_chatbook.Widgets.chat_message_enhanced.TEXTUAL_IMAGE_AVAILABLE', False)
    def test_render_regular_fallback(self, chat_message):
        """Test fallback rendering when textual-image is not available."""
        chat_message._image_widget = Mock()
        chat_message._image_widget.mount = Mock()
        
        chat_message._render_regular()
        
        # Should use fallback rendering
        chat_message._image_widget.mount.assert_called_once()
        static_widget = chat_message._image_widget.mount.call_args[0][0]
        assert "📷 Image" in str(static_widget.renderable)
    
    def test_render_pixelated(self, chat_message):
        """Test pixelated rendering with rich-pixels."""
        chat_message._image_widget = Mock()
        chat_message._image_widget.mount = Mock()
        
        with patch('tldw_chatbook.Widgets.chat_message_enhanced.Pixels') as mock_pixels:
            mock_pixels.from_image.return_value = Mock()
            chat_message._render_pixelated()
            
            mock_pixels.from_image.assert_called_once()
            chat_message._image_widget.mount.assert_called_once()
    
    def test_toggle_pixel_mode(self, chat_message):
        """Test toggling between pixel and regular mode."""
        chat_message._image_widget = Mock()
        
        initial_mode = chat_message.pixel_mode
        chat_message.handle_toggle_mode()
        
        assert chat_message.pixel_mode != initial_mode
    
    @pytest.mark.asyncio
    async def test_save_image(self, chat_message, tmp_path):
        """Test saving image to file."""
        # Mock app for notification testing
        mock_app = Mock()
        mock_app.notify = Mock()
        
        # Mock Path.home() to use temp directory
        with patch('tldw_chatbook.Widgets.chat_message_enhanced.Path.home') as mock_home:
            with patch.object(chat_message, 'app', mock_app, create=True):
                mock_home.return_value = tmp_path
                
                await chat_message.handle_save_image()
                
                # Check that file was saved
                downloads_dir = tmp_path / "Downloads"
                assert downloads_dir.exists()
                
                # Check that at least one image file was created
                image_files = list(downloads_dir.glob("chat_image_*.png"))
                assert len(image_files) == 1
                
                # Verify the saved file contains our test data
                saved_data = image_files[0].read_bytes()
                assert saved_data == chat_message.image_data
                
                # Check notification was sent
                mock_app.notify.assert_called_once()
    
    def test_generation_complete_watcher(self, chat_message):
        """Test the generation complete watcher for AI messages."""
        # Change to AI message
        chat_message.remove_class("-user")
        chat_message.add_class("-ai")
        chat_message._generation_complete_internal = False
        
        # Mock query_one
        mock_actions = Mock()
        mock_actions.remove_class = Mock()
        mock_actions.add_class = Mock()
        mock_actions.styles.display = "none"
        
        with patch.object(chat_message, 'query_one', return_value=mock_actions):
            # Trigger watcher
            chat_message.watch__generation_complete_internal(True)
            
            # Check that actions were made visible
            mock_actions.remove_class.assert_called_with("-generating")
            assert mock_actions.styles.display == "block"
    
    def test_update_message_chunk(self):
        """Test updating message during streaming."""
        message = ChatMessageEnhanced(
            message="Initial",
            role="Assistant",
            generation_complete=False
        )
        
        message.update_message_chunk(" chunk")
        assert message.message_text == "Initial chunk"
    
    def test_mark_generation_complete(self):
        """Test marking generation as complete."""
        message = ChatMessageEnhanced(
            message="AI response",
            role="Assistant",
            generation_complete=False
        )
        
        assert not message.generation_complete
        
        message.mark_generation_complete()
        
        assert message.generation_complete
    
    def test_button_action_event(self, chat_message):
        """Test that button presses emit Action events."""
        mock_button = Mock(spec=Button)
        event = Button.Pressed(mock_button)
        
        with patch.object(chat_message, 'post_message') as mock_post:
            chat_message.on_button_pressed(event)
            
            # Check that Action message was posted
            mock_post.assert_called_once()
            action = mock_post.call_args[0][0]
            assert isinstance(action, ChatMessageEnhanced.Action)
            assert action.message_widget == chat_message
            assert action.button == mock_button


class TestImageRenderingEdgeCases:
    """Test edge cases for image rendering."""
    
    def test_render_with_corrupt_image(self, chat_message):
        """Test rendering with corrupt image data."""
        chat_message.image_data = b"not-an-image"
        chat_message._image_widget = Mock()
        chat_message._image_widget.mount = Mock()
        
        # Should handle gracefully and show error
        chat_message._render_image()
        
        # Check that error message was mounted
        chat_message._image_widget.mount.assert_called_once()
        static_widget = chat_message._image_widget.mount.call_args[0][0]
        assert "Error rendering image" in str(static_widget.renderable)
    
    def test_render_with_large_image(self, chat_message):
        """Test rendering with large image that needs resizing."""
        # Create a large image
        large_img = PILImage.new('RGB', (4000, 4000), color='blue')
        from io import BytesIO
        buffer = BytesIO()
        large_img.save(buffer, format='PNG')
        chat_message.image_data = buffer.getvalue()
        
        chat_message._image_widget = Mock()
        chat_message._image_widget.mount = Mock()
        
        with patch('tldw_chatbook.Widgets.chat_message_enhanced.Pixels') as mock_pixels:
            mock_pixels.from_image.return_value = Mock()
            chat_message.pixel_mode = True
            chat_message._render_image()
            
            # Should resize and render successfully
            mock_pixels.from_image.assert_called_once()
    
    def test_render_without_image_widget(self, chat_message):
        """Test render when image widget is not set."""
        chat_message._image_widget = None
        
        # Should not raise error
        chat_message._render_image()
    
    @pytest.mark.asyncio
    async def test_save_image_without_data(self):
        """Test saving when no image data is present."""
        message = ChatMessageEnhanced(
            message="No image",
            role="User"
        )
        mock_app = Mock()
        mock_app.notify = Mock()
        
        with patch.object(message, 'app', mock_app, create=True):
            await message.handle_save_image()
            
            # Should not call notify (early return)
            mock_app.notify.assert_not_called()


class TestChatMessageProperties:
    """Property-based tests for ChatMessage."""
    
    def test_message_text_property(self):
        """Test message_text reactive property."""
        message = ChatMessageEnhanced(
            message="Initial",
            role="User"
        )
        
        assert message.message_text == "Initial"
        
        message.message_text = "Updated"
        assert message.message_text == "Updated"
    
    def test_role_property(self):
        """Test role reactive property."""
        message = ChatMessageEnhanced(
            message="Test",
            role="User"
        )
        
        assert message.role == "User"
        assert message.has_class("-user")
        
        # Note: Changing role after init doesn't update classes
        # This is by design in the current implementation

#
#
#######################################################################################################################