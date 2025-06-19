# Tests/Widgets/test_chat_message_enhanced_async.py
# Description: Async unit tests for ChatMessageEnhanced widget using proper Textual patterns
#
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from io import BytesIO

from PIL import Image as PILImage
from textual.app import App, ComposeResult
from textual.widgets import Button, Static
from textual.containers import Vertical, Horizontal

from tldw_chatbook.Widgets.chat_message_enhanced import ChatMessageEnhanced

#
# Test Fixtures
#

@pytest.fixture
def sample_image_data():
    """Create sample image data for testing."""
    img = PILImage.new('RGB', (100, 100), color='red')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


class ChatMessageTestApp(App):
    """Test app for ChatMessageEnhanced widget."""
    
    def __init__(self, message_kwargs=None):
        super().__init__()
        self.message_kwargs = message_kwargs or {}
    
    def compose(self) -> ComposeResult:
        """Create child widgets."""
        yield ChatMessageEnhanced(**self.message_kwargs)


#
# Async Unit Tests
#

class TestChatMessageEnhancedAsync:
    """Test suite for ChatMessageEnhanced widget using async patterns."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, sample_image_data):
        """Test that ChatMessageEnhanced initializes correctly."""
        app = ChatMessageTestApp({
            "message": "Hello, world!",
            "role": "User",
            "image_data": sample_image_data,
            "image_mime_type": "image/png"
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            assert message.message_text == "Hello, world!"
            assert message.role == "User"
            assert message.image_data == sample_image_data
            assert message.image_mime_type == "image/png"
            assert message.pixel_mode is False
            assert message.has_class("-user")
    
    @pytest.mark.asyncio
    async def test_ai_message_initialization(self):
        """Test AI message initialization."""
        app = ChatMessageTestApp({
            "message": "AI response",
            "role": "Assistant",
            "generation_complete": False
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            assert message.role == "Assistant"
            assert message.generation_complete is False
            assert message.has_class("-ai")
            assert not message.has_class("-user")
    
    @pytest.mark.asyncio
    async def test_compose_with_image(self, sample_image_data):
        """Test that compose includes image elements when image data is present."""
        app = ChatMessageTestApp({
            "message": "Test message",
            "role": "User",
            "image_data": sample_image_data,
            "image_mime_type": "image/png"
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            # Check that the widget is properly composed
            assert message.is_mounted
            assert len(message.children) > 0
            
            # Should have vertical container
            verticals = message.query(Vertical)
            assert len(verticals) > 0
    
    @pytest.mark.asyncio
    async def test_compose_without_image(self):
        """Test compose method without image data."""
        app = ChatMessageTestApp({
            "message": "Text only message",
            "role": "User"
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            # Check basic structure
            assert message.is_mounted
            assert len(message.children) > 0
    
    @patch('tldw_chatbook.Widgets.chat_message_enhanced.textual_image')
    @pytest.mark.asyncio
    async def test_render_regular_with_textual_image(self, mock_textual_image, sample_image_data):
        """Test regular rendering when textual-image is available."""
        mock_image_widget = Mock()
        mock_textual_image.return_value = mock_image_widget
        
        app = ChatMessageTestApp({
            "message": "Test with image",
            "role": "User",
            "image_data": sample_image_data,
            "image_mime_type": "image/png"
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            # Force a render
            await pilot.pause()
            
            # The render method should have been called
            assert message.is_mounted
    
    @pytest.mark.asyncio
    async def test_toggle_pixel_mode(self, sample_image_data):
        """Test toggling between regular and pixel rendering modes."""
        app = ChatMessageTestApp({
            "message": "Test message",
            "role": "User",
            "image_data": sample_image_data,
            "image_mime_type": "image/png"
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            initial_mode = message.pixel_mode
            message.toggle_pixel_mode()
            assert message.pixel_mode != initial_mode
            
            message.toggle_pixel_mode()
            assert message.pixel_mode == initial_mode
    
    @pytest.mark.asyncio
    async def test_save_image(self, sample_image_data, tmp_path):
        """Test saving image to file."""
        app = ChatMessageTestApp({
            "message": "Test message",
            "role": "User",
            "image_data": sample_image_data,
            "image_mime_type": "image/png"
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            # Mock the app's notify method
            app.notify = Mock()
            
            # Test save functionality
            save_path = tmp_path / "test_image.png"
            with patch('pathlib.Path.open', create=True) as mock_open:
                mock_file = Mock()
                mock_open.return_value.__enter__.return_value = mock_file
                
                message.save_image(str(save_path))
                
                mock_file.write.assert_called_once_with(sample_image_data)
                app.notify.assert_called()
    
    @pytest.mark.asyncio
    async def test_update_message_chunk(self):
        """Test updating message content in chunks."""
        app = ChatMessageTestApp({
            "message": "Initial",
            "role": "Assistant",
            "generation_complete": False
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            message.update_message_chunk(" chunk")
            assert message.message_text == "Initial chunk"
            
            message.update_message_chunk(" more")
            assert message.message_text == "Initial chunk more"
    
    @pytest.mark.asyncio
    async def test_mark_generation_complete(self):
        """Test marking generation as complete."""
        app = ChatMessageTestApp({
            "message": "Test",
            "role": "Assistant",
            "generation_complete": False
        })
        
        async with app.run_test() as pilot:
            message = app.query_one(ChatMessageEnhanced)
            
            assert message.generation_complete is False
            
            message.mark_generation_complete()
            assert message.generation_complete is True


class TestChatMessageProperties:
    """Test ChatMessageEnhanced properties without async context."""
    
    def test_message_text_property(self):
        """Test message_text property getter/setter."""
        message = ChatMessageEnhanced(message="Initial text", role="User")
        
        assert message.message_text == "Initial text"
        
        message.message_text = "Updated text"
        assert message.message_text == "Updated text"
    
    def test_role_property(self):
        """Test role property and CSS class updates."""
        message = ChatMessageEnhanced(message="Test", role="User")
        
        assert message.role == "User"
        assert message.has_class("-user")
        
        message.role = "Assistant"
        assert message.role == "Assistant"