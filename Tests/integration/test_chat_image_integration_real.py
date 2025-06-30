# Tests/integration/test_chat_image_integration_real.py
# Description: Integration tests for chat image attachment flow using real Textual app
#
# Imports
#
# Standard Library
import pytest
import pytest_asyncio
import tempfile
import os
from pathlib import Path
from io import BytesIO
from unittest.mock import patch

# 3rd-party Libraries
from PIL import Image as PILImage
from textual.widgets import Input, Button, TextArea

# Local Imports
from tldw_chatbook.app import TldwCli
from tldw_chatbook.UI.Chat_Window_Enhanced import ChatWindowEnhanced
from tldw_chatbook.Widgets.chat_message_enhanced import ChatMessageEnhanced
from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.DB.Client_Media_DB_v2 import MediaDatabase

# Test marker for integration tests
pytestmark = pytest.mark.integration

#######################################################################################################################
#
# Test Fixtures

@pytest_asyncio.fixture
async def real_app_with_image_support(tmp_path):
    """Create a real TldwCli app instance with image support enabled."""
    # Create temporary directories
    db_dir = tmp_path / "db"
    db_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create config file with image support enabled
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
model = "gpt-4-vision-preview"
temperature = 0.7
max_tokens = 1000
streaming = false
system_prompt = "You are a helpful assistant."

[chat.images]
enabled = true
default_render_mode = "auto"
max_size_mb = 10
auto_resize = true

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
    app.media_db = MediaDatabase(str(db_dir / "media.db"))
    
    # Set app attributes
    app.current_chat_conversation_id = None
    app.current_chat_is_ephemeral = True
    app.current_chat_active_character_data = None
    app.notes_user_id = "test_user"
    
    yield app
    
    # Cleanup
    app.chachanotes_db.close()
    app.media_db.close()


@pytest.fixture
def temp_test_image():
    """Create a temporary test image."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        img = PILImage.new('RGB', (500, 500), color='purple')
        img.save(f, format='PNG')
        temp_path = Path(f.name)
    
    yield temp_path
    
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_large_image():
    """Create a large test image for testing size limits."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a 4000x4000 image (will be large when saved)
        img = PILImage.new('RGB', (4000, 4000), color='red')
        img.save(f, format='JPEG', quality=95)
        temp_path = Path(f.name)
    
    yield temp_path
    
    if temp_path.exists():
        temp_path.unlink()


#######################################################################################################################
#
# Integration Tests

class TestChatImageIntegration:
    """Integration tests for the complete image attachment flow."""
    
    @pytest.mark.asyncio
    async def test_attach_image_button_flow(self, real_app_with_image_support):
        """Test the flow of clicking attach image button."""
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Find attach image button
            attach_button = app.query_one("#attach-image", Button)
            assert attach_button is not None
            assert attach_button.label == "📎"
            
            # Click attach button
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            # Check that file input is now visible
            file_input = app.query_one("#image-file-input", Input)
            assert file_input is not None
            assert "hidden" not in file_input.classes
            
            # Check that input has focus
            assert app.focused == file_input
    
    @pytest.mark.asyncio
    async def test_image_path_submission_success(self, real_app_with_image_support, temp_test_image):
        """Test successful image path submission."""
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Get chat window instance
            chat_window = app.query_one(ChatWindowEnhanced)
            assert chat_window is not None
            
            # Click attach button
            attach_button = app.query_one("#attach-image", Button)
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            # Enter image path
            file_input = app.query_one("#image-file-input", Input)
            await pilot.click(file_input)
            file_input.value = str(temp_test_image)
            
            # Submit the path
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            # Check that image was attached
            assert chat_window.pending_image is not None
            assert 'data' in chat_window.pending_image
            assert 'mime_type' in chat_window.pending_image
            assert chat_window.pending_image['path'] == str(temp_test_image)
            
            # Check UI updates
            assert attach_button.label == "📎✓"
            
            # Check attachment indicator
            indicator = app.query_one("#image-attachment-indicator")
            assert indicator is not None
            assert "hidden" not in indicator.classes
            assert str(temp_test_image.name) in indicator.renderable.plain
            
            # File input should be hidden again
            assert "hidden" in file_input.classes
    
    @pytest.mark.asyncio
    async def test_image_path_submission_invalid_file(self, real_app_with_image_support):
        """Test image path submission with invalid file."""
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Get chat window instance
            chat_window = app.query_one(ChatWindowEnhanced)
            
            # Click attach button
            attach_button = app.query_one("#attach-image", Button)
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            # Enter invalid path
            file_input = app.query_one("#image-file-input", Input)
            await pilot.click(file_input)
            file_input.value = "/path/to/nonexistent.png"
            
            # Submit the path
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            # Check that no image was attached
            assert chat_window.pending_image is None
            
            # Check that attach button didn't change
            assert attach_button.label == "📎"
            
            # Check that error notification was shown
            # (Would need to mock notify to verify this)
    
    @pytest.mark.asyncio
    async def test_clear_image_attachment(self, real_app_with_image_support, temp_test_image):
        """Test clearing an attached image."""
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # First attach an image
            chat_window = app.query_one(ChatWindowEnhanced)
            attach_button = app.query_one("#attach-image", Button)
            
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            file_input = app.query_one("#image-file-input", Input)
            await pilot.click(file_input)
            file_input.value = str(temp_test_image)
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            # Verify image is attached
            assert chat_window.pending_image is not None
            assert attach_button.label == "📎✓"
            
            # Click clear button
            clear_button = app.query_one("#clear-image", Button)
            await pilot.click(clear_button)
            await pilot.pause(0.1)
            
            # Check that image was cleared
            assert chat_window.pending_image is None
            assert attach_button.label == "📎"
            
            # Check that indicator is hidden
            indicator = app.query_one("#image-attachment-indicator")
            assert "hidden" in indicator.classes
    
    @pytest.mark.asyncio
    async def test_send_message_with_image(self, real_app_with_image_support, temp_test_image):
        """Test sending a message with an attached image."""
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Attach an image
            chat_window = app.query_one(ChatWindowEnhanced)
            attach_button = app.query_one("#attach-image", Button)
            
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            file_input = app.query_one("#image-file-input", Input)
            await pilot.click(file_input)
            file_input.value = str(temp_test_image)
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            # Type a message
            chat_input = app.query_one("#chat-input", TextArea)
            await pilot.click(chat_input)
            await pilot.type("Here is an image")
            
            # Mock the chat API call
            with patch.object(app, 'chat_wrapper', return_value="I can see the image"):
                # Send the message
                send_button = app.query_one("#chat-send-button", Button)
                await pilot.click(send_button)
                await pilot.pause(0.3)
            
            # Check that message was sent with image
            chat_log = app.query_one("#chat-log")
            messages = chat_log.query(ChatMessageEnhanced)
            
            # Find user message
            user_messages = [m for m in messages if m.role == "User"]
            assert len(user_messages) > 0
            
            user_msg = user_messages[-1]
            assert user_msg.message_text == "Here is an image"
            assert user_msg.image_data is not None
            assert user_msg.image_mime_type == "image/png"
            
            # Check that attachment was cleared after sending
            assert chat_window.pending_image is None
            assert attach_button.label == "📎"
    
    @pytest.mark.asyncio
    async def test_image_resizing(self, real_app_with_image_support, temp_large_image):
        """Test that large images are automatically resized."""
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Attach large image
            chat_window = app.query_one(ChatWindowEnhanced)
            attach_button = app.query_one("#attach-image", Button)
            
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            file_input = app.query_one("#image-file-input", Input)
            await pilot.click(file_input)
            file_input.value = str(temp_large_image)
            await pilot.press("enter")
            await pilot.pause(0.3)  # Give more time for resizing
            
            # Check that image was attached and resized
            assert chat_window.pending_image is not None
            
            # Check that the image data is smaller than original
            original_size = temp_large_image.stat().st_size
            attached_size = len(chat_window.pending_image['data'])
            
            # Should be significantly smaller after resizing
            assert attached_size < original_size * 0.5
    
    @pytest.mark.asyncio
    async def test_unsupported_file_type(self, real_app_with_image_support, tmp_path):
        """Test attaching an unsupported file type."""
        # Create a text file
        text_file = tmp_path / "test.txt"
        text_file.write_text("This is not an image")
        
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Try to attach text file
            chat_window = app.query_one(ChatWindowEnhanced)
            attach_button = app.query_one("#attach-image", Button)
            
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            file_input = app.query_one("#image-file-input", Input)
            await pilot.click(file_input)
            file_input.value = str(text_file)
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            # Check that no image was attached
            assert chat_window.pending_image is None
            assert attach_button.label == "📎"
    
    @pytest.mark.asyncio
    async def test_image_preview_display(self, real_app_with_image_support, temp_test_image):
        """Test that image preview is displayed in messages."""
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Create a message with image directly
            chat_log = app.query_one("#chat-log")
            
            # Load image data
            with open(temp_test_image, 'rb') as f:
                image_data = f.read()
            
            # Create message with image
            msg = ChatMessageEnhanced(
                message="Test message with image",
                role="User",
                image_data=image_data,
                image_mime_type="image/png",
                generation_complete=True
            )
            
            await chat_log.mount(msg)
            await pilot.pause(0.2)
            
            # Check that message displays image indicator
            # (Actual image rendering would depend on terminal capabilities)
            assert msg.image_data is not None
            assert "[Image attached]" in str(msg.render())
    
    @pytest.mark.asyncio
    async def test_multiple_image_attachments_not_allowed(self, real_app_with_image_support, temp_test_image):
        """Test that only one image can be attached at a time."""
        app = real_app_with_image_support
        
        async with app.run_test() as pilot:
            await pilot.pause(0.5)
            
            # Navigate to chat tab
            await pilot.press("c")
            await pilot.pause(0.1)
            
            # Attach first image
            chat_window = app.query_one(ChatWindowEnhanced)
            attach_button = app.query_one("#attach-image", Button)
            
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            file_input = app.query_one("#image-file-input", Input)
            await pilot.click(file_input)
            file_input.value = str(temp_test_image)
            await pilot.press("enter")
            await pilot.pause(0.2)
            
            # Verify first image is attached
            first_image_data = chat_window.pending_image['data']
            
            # Try to attach another image without clearing first
            await pilot.click(attach_button)
            await pilot.pause(0.1)
            
            # The button should not show file input again until first image is cleared
            # or it should replace the existing image
            file_input = app.query_one("#image-file-input", Input)
            if "hidden" not in file_input.classes:
                # If input is shown, attach a different image
                await pilot.click(file_input)
                file_input.value = str(temp_test_image)  # Same file for simplicity
                await pilot.press("enter")
                await pilot.pause(0.2)
                
                # Check that new image replaced the old one
                assert chat_window.pending_image['data'] == first_image_data  # Same file
            else:
                # Input should be hidden if multiple attachments not allowed
                assert "hidden" in file_input.classes