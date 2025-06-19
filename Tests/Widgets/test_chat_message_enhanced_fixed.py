# Tests/Widgets/test_chat_message_enhanced_fixed.py
# Description: Fixed unit tests for the enhanced ChatMessage widget with proper Textual patterns
#
# Imports
#
# Standard Library
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio

# 3rd-party Libraries
from PIL import Image as PILImage
from textual.app import App
from textual.widgets import Button
from textual import _context

# Local Imports
from tldw_chatbook.Widgets.chat_message_enhanced import ChatMessageEnhanced

#
#######################################################################################################################
#
# Test Fixtures

class TestApp(App):
    """Minimal test app for widget testing."""
    def compose(self):
        yield ChatMessageEnhanced(
            message="Test message",
            role="User",
            generation_complete=True,
            message_id="test-123",
            message_version=1,
            timestamp="2024-01-20 10:00:00"
        )

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
async def app_with_message(sample_image_data):
    """Create a ChatMessageEnhanced instance within a proper app context."""
    app = TestApp()
    async with app.run_test() as pilot:
        # Get the message widget from the app
        message = app.query_one(ChatMessageEnhanced)
        message.image_data = sample_image_data
        message.image_mime_type = "image/png"
        yield message, pilot, app


#
# Unit Tests
#

class TestChatMessageEnhancedAsync:
    """Test suite for ChatMessageEnhanced widget using async patterns."""
    
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
    
    @pytest.mark.asyncio
    async def test_compose_with_image(self, app_with_message):
        """Test compose method with image data."""
        message, pilot, app = await app_with_message.__anext__()
        
        # The widget should be composed and mounted
        assert message.is_mounted
        assert message.image_data is not None
        
    @pytest.mark.asyncio
    async def test_toggle_pixel_mode(self, app_with_message):
        """Test toggling pixel mode."""
        message, pilot, app = await app_with_message.__anext__()
        
        initial_mode = message.pixel_mode
        message.toggle_pixel_mode()
        assert message.pixel_mode != initial_mode
        
        message.toggle_pixel_mode()
        assert message.pixel_mode == initial_mode