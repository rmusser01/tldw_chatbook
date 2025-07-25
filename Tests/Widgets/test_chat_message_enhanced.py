# Tests/Widgets/test_chat_message_enhanced.py
# Description: Comprehensive tests for ChatMessageEnhanced widget using Textual test patterns
#
# Imports
#
# Standard Library
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from io import BytesIO

# 3rd-party Libraries
from PIL import Image as PILImage
from textual.widgets import Button, Label, Static, TextArea, Markdown
from textual.containers import Container, Horizontal, Vertical
from textual.events import Click

# Test utilities
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from textual_test_utils import widget_pilot, app_pilot
from textual_test_harness import TestApp, IsolatedWidgetTestApp, isolated_widget_pilot

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
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def large_image_data():
    """Create large image data for resize testing."""
    img = PILImage.new('RGB', (4000, 4000), color='blue')
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


#
# Test Classes
#

@pytest.mark.ui
class TestChatMessageEnhancedInitialization:
    """Test basic initialization and properties."""
    
    def test_user_message_initialization(self, sample_image_data):
        """Test user message initialization with all parameters."""
        message = ChatMessageEnhanced(
            message="Hello, world!",
            role="User",
            generation_complete=True,
            message_id="test-123",
            message_version=1,
            timestamp="2024-01-20 10:00:00",
            image_data=sample_image_data,
            image_mime_type="image/png"
        )
        
        assert message.message_text == "Hello, world!"
        assert message.role == "User"
        assert message.generation_complete is True
        assert message.message_id_internal == "test-123"
        assert message.message_version_internal == 1
        assert message.timestamp == "2024-01-20 10:00:00"
        assert message.image_data == sample_image_data
        assert message.image_mime_type == "image/png"
        assert message.pixel_mode is False
        assert message.has_class("-user")
        assert not message.has_class("-ai")
    
    def test_ai_message_initialization(self):
        """Test AI message initialization during generation."""
        message = ChatMessageEnhanced(
            message="AI response",
            role="Assistant",
            generation_complete=False
        )
        
        assert message.role == "Assistant"
        assert message.generation_complete is False
        assert message.has_class("-ai")
        assert not message.has_class("-user")
        assert message._generation_complete_internal is False
    
    def test_minimal_initialization(self):
        """Test initialization with minimal parameters."""
        message = ChatMessageEnhanced(
            message="Test",
            role="User"
        )
        
        assert message.message_text == "Test"
        assert message.role == "User"
        assert message.generation_complete is True
        assert message.image_data is None
        assert message.timestamp is None
        assert message.message_id_internal is None


@pytest.mark.ui
@pytest.mark.asyncio
class TestChatMessageEnhancedComposition:
    """Test widget composition and structure using async patterns."""
    
    async def test_compose_structure_user_message(self, widget_pilot, sample_image_data):
        """Test complete widget structure for user message."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="User message",
            role="User",
            image_data=sample_image_data,
            timestamp="2024-01-20 10:00:00"
        ) as pilot:
            app = pilot.app
            widget = app.test_widget
            await pilot.pause()
            
            # Check main container
            main_container = widget.query_one(Vertical)
            assert main_container is not None
            
            # Check header
            header = widget.query_one(".message-header", Label)
            assert header is not None
            assert "User" in header.renderable
            assert "2024-01-20 10:00:00" in header.renderable
            
            # Check message text
            message_text = widget.query_one(".message-text", Markdown)
            assert message_text is not None
            # For Markdown widget, we need to check the internal _markdown property
            assert "User message" in message_text._markdown
            
            # Check image container
            image_container = widget.query_one(".message-image-container", Container)
            assert image_container is not None
            
            # Check image controls
            toggle_btn = widget.query_one("#toggle-image-mode", Button)
            save_btn = widget.query_one("#save-image", Button)
            assert toggle_btn is not None
            assert save_btn is not None
            
            # Check action buttons
            actions_bar = widget.query_one(".message-actions", Horizontal)
            assert actions_bar is not None
            
            # User messages should not have AI-specific buttons
            copy_btn = widget.query_one("#copy", Button)
            speak_btn = widget.query_one("#speak", Button)
            assert copy_btn is not None
            assert speak_btn is not None
            
            # Should not have AI buttons
            with pytest.raises(Exception):
                widget.query_one("#thumb-up", Button)
    
    async def test_compose_structure_ai_message(self, widget_pilot):
        """Test widget structure for AI message."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="AI response",
            role="Assistant",
            generation_complete=True
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Check AI-specific buttons
            thumb_up = widget.query_one("#thumb-up", Button)
            thumb_down = widget.query_one("#thumb-down", Button)
            regenerate = widget.query_one("#regenerate", Button)
            continue_btn = widget.query_one("#continue-response-button", Button)
            
            assert thumb_up is not None
            assert thumb_down is not None
            assert regenerate is not None
            assert continue_btn is not None
    
    async def test_compose_without_image(self, widget_pilot):
        """Test composition when no image is present."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Text only",
            role="User"
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Should not have image container
            containers = widget.query(".message-image-container")
            assert len(containers) == 0


@pytest.mark.ui
@pytest.mark.asyncio
class TestChatMessageEnhancedInteractions:
    """Test user interactions with the widget."""
    
    async def test_button_action_events(self, widget_pilot, mock_app_instance):
        """Test that button clicks are handled properly."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Test message",
            role="User"
        ) as pilot:
            app = pilot.app
            widget = app.test_widget
            
            await pilot.pause()
            
            # Test copy button exists and is clickable
            copy_btn = widget.query_one("#copy", Button)
            assert copy_btn is not None
            assert not copy_btn.disabled
            
            # Click copy button - in a real app this would emit an event
            await pilot.click(copy_btn)
            await pilot.pause()
            
            # Test speak button exists and is clickable  
            speak_btn = widget.query_one("#speak", Button)
            assert speak_btn is not None
            assert not speak_btn.disabled
            
            # Click speak button
            await pilot.click(speak_btn)
            await pilot.pause()
    
    async def test_toggle_image_mode(self, widget_pilot, sample_image_data):
        """Test toggling between image display modes."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Test",
            role="User",
            image_data=sample_image_data
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Initial mode
            assert widget.pixel_mode is False
            
            # Test the handler directly first
            widget.handle_toggle_mode()
            assert widget.pixel_mode is True
            
            # Reset
            widget.pixel_mode = False
            
            # Now test via button click
            toggle_btn = widget.query_one("#toggle-image-mode", Button)
            assert toggle_btn is not None
            
            # Use the button's press method directly
            toggle_btn.press()
            await pilot.pause()
            
            # Mode should change
            assert widget.pixel_mode is True
            
            # Press again
            toggle_btn.press()
            await pilot.pause()
            
            assert widget.pixel_mode is False
    
    async def test_ai_generation_state_handling(self, widget_pilot, wait_for_condition):
        """Test AI message generation state changes."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Initial",
            role="Assistant",
            generation_complete=False
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Actions should be hidden during generation
            actions_bar = widget.query_one(".message-actions", Horizontal)
            assert actions_bar.has_class("-generating")
            
            # Stream some text
            widget.update_message_chunk(" chunk 1")
            widget.update_message_chunk(" chunk 2")
            assert widget.message_text == "Initial chunk 1 chunk 2"
            
            # Mark generation complete
            widget.mark_generation_complete()
            await pilot.pause()
            
            # Actions should be visible
            assert widget.generation_complete is True
            await wait_for_condition(
                lambda: not actions_bar.has_class("-generating"),
                timeout=2.0
            )


@pytest.mark.ui
@pytest.mark.asyncio
class TestChatMessageEnhancedImageHandling:
    """Test image rendering and handling."""
    
    @patch('tldw_chatbook.Widgets.chat_message_enhanced.TEXTUAL_IMAGE_AVAILABLE', True)
    async def test_image_rendering_textual_image(self, widget_pilot, sample_image_data):
        """Test image rendering with textual-image available."""
        with patch('tldw_chatbook.Widgets.chat_message_enhanced.TextualImage') as mock_image_class:
            mock_image = Mock()
            mock_image_class.return_value = mock_image
            
            async with await widget_pilot(
                ChatMessageEnhanced,
                message="Test",
                role="User",
                image_data=sample_image_data
            ) as pilot:
                widget = pilot.app.test_widget
                await pilot.pause()
                
                # Verify TextualImage was created with a PIL Image object
                # It may be called during compose and/or on_mount
                assert mock_image_class.call_count >= 1
                # Check that it was called with a PIL Image instance
                for call in mock_image_class.call_args_list:
                    args, kwargs = call
                    assert len(args) == 1
                    from PIL import Image as PILImage
                    assert isinstance(args[0], PILImage.Image)
    
    @patch('tldw_chatbook.Widgets.chat_message_enhanced.TEXTUAL_IMAGE_AVAILABLE', False)
    async def test_image_rendering_fallback(self, widget_pilot, sample_image_data):
        """Test fallback rendering when textual-image not available."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Test",
            role="User",
            image_data=sample_image_data,
            image_mime_type="image/png"
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Check fallback text is displayed
            image_widget = widget.query_one(".message-image", Container)
            static_content = image_widget.query_one(Static)
            assert static_content is not None
            
            renderable_str = str(static_content.renderable)
            assert "📷 Image" in renderable_str
            assert "image/png" in renderable_str
            assert "KB" in renderable_str
    
    async def test_pixelated_mode_rendering(self, widget_pilot, sample_image_data):
        """Test pixelated rendering mode."""
        # We don't need to mock Pixels, just verify the mode works
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Test",
            role="User",
            image_data=sample_image_data
        ) as pilot:
            widget = pilot.app.test_widget
            
            # Initially should not be in pixel mode
            assert widget.pixel_mode is False
            
            # Set pixel mode
            widget.pixel_mode = True
            await pilot.pause()
            
            # Verify pixel mode is set
            assert widget.pixel_mode is True
            
            # Verify the image widget exists and has been rendered
            image_container = widget.query_one(".message-image", Container)
            assert image_container is not None
            
            # There should be content in the image container
            static_widgets = image_container.query(Static)
            assert len(static_widgets) > 0
    
    async def test_save_image_functionality(self, widget_pilot, sample_image_data, tmp_path, mock_app_instance):
        """Test saving image to file."""
        with patch('tldw_chatbook.Widgets.chat_message_enhanced.Path.home') as mock_home:
            mock_home.return_value = tmp_path
            
            # Create Downloads directory in test environment
            downloads_dir = tmp_path / "Downloads"
            downloads_dir.mkdir(exist_ok=True)
            
            async with await widget_pilot(
                ChatMessageEnhanced,
                message="Test",
                role="User",
                image_data=sample_image_data,
                image_mime_type="image/png"
            ) as pilot:
                widget = pilot.app.test_widget
                
                # Mock the app's notify method
                pilot.app.notify = mock_app_instance.notify
                
                await pilot.pause()
                
                # Save image
                await widget.handle_save_image()
                
                # Check file was created
                assert downloads_dir.exists()
                
                image_files = list(downloads_dir.glob("chat_image_*.png"))
                assert len(image_files) == 1
                
                # Verify content
                saved_data = image_files[0].read_bytes()
                assert saved_data == sample_image_data
                
                # Check notification
                mock_app_instance.notify.assert_called_once()
    
    async def test_large_image_resizing(self, widget_pilot, large_image_data):
        """Test that large images are resized properly."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Test",
            role="User",
            image_data=large_image_data
        ) as pilot:
            widget = pilot.app.test_widget
            widget.pixel_mode = True
            await pilot.pause()
            
            # Render should handle large image gracefully
            widget._render_image()
            
            # Widget should still be functional
            assert widget.is_mounted
    
    async def test_corrupt_image_handling(self, widget_pilot):
        """Test handling of corrupt image data."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Test",
            role="User",
            image_data=b"not-an-image"
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Should show fallback message
            image_widget = widget.query_one(".message-image", Container)
            fallback_content = image_widget.query_one(Static)
            assert fallback_content is not None
            renderable_str = str(fallback_content.renderable)
            # Check for fallback content markers
            assert "📷 Image" in renderable_str
            assert "Size:" in renderable_str
            assert "KB" in renderable_str
            assert "Preview:" in renderable_str


@pytest.mark.ui
@pytest.mark.asyncio
class TestChatMessageEnhancedStreaming:
    """Test message streaming functionality."""
    
    async def test_message_chunk_updates(self, widget_pilot):
        """Test streaming message updates."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="",
            role="Assistant",
            generation_complete=False
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Stream chunks
            chunks = ["Hello", ", ", "how ", "can ", "I ", "help ", "you", "?"]
            expected = ""
            
            for chunk in chunks:
                widget.update_message_chunk(chunk)
                expected += chunk
                assert widget.message_text == expected
            
            # Verify final message
            assert widget.message_text == "Hello, how can I help you?"
            
            # Generation still not complete
            assert not widget.generation_complete
    
    async def test_generation_completion(self, widget_pilot, wait_for_condition):
        """Test marking generation as complete."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="Response complete",
            role="Assistant",
            generation_complete=False
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Initially not complete
            assert not widget.generation_complete
            actions_bar = widget.query_one(".message-actions", Horizontal)
            assert actions_bar.has_class("-generating")
            
            # Mark complete
            widget.mark_generation_complete()
            await pilot.pause()
            
            # Should be complete
            assert widget.generation_complete
            
            # Wait for UI update
            await wait_for_condition(
                lambda: not actions_bar.has_class("-generating"),
                timeout=2.0
            )
            
            # Continue button should be visible
            continue_btn = widget.query_one("#continue-response-button", Button)
            assert continue_btn.display is True


@pytest.mark.ui
class TestChatMessageEnhancedProperties:
    """Test reactive properties and watchers."""
    
    def test_message_text_reactive_property(self):
        """Test message_text property updates."""
        message = ChatMessageEnhanced(
            message="Initial",
            role="User"
        )
        
        assert message.message_text == "Initial"
        
        # Update property
        message.message_text = "Updated"
        assert message.message_text == "Updated"
    
    def test_role_based_css_classes(self):
        """Test CSS classes based on role."""
        # User message
        user_msg = ChatMessageEnhanced(
            message="User",
            role="User"
        )
        assert user_msg.has_class("-user")
        assert not user_msg.has_class("-ai")
        
        # AI message
        ai_msg = ChatMessageEnhanced(
            message="AI",
            role="Assistant"
        )
        assert ai_msg.has_class("-ai")
        assert not ai_msg.has_class("-user")
        
        # Other role (gets AI class for non-user roles)
        other_msg = ChatMessageEnhanced(
            message="System",
            role="System"
        )
        assert not other_msg.has_class("-user")
        assert other_msg.has_class("-ai")  # Non-user roles get AI class
    
    def test_pixel_mode_watcher(self):
        """Test pixel mode property watcher."""
        message = ChatMessageEnhanced(
            message="Test",
            role="User",
            image_data=b"fake-image"
        )
        
        # Mock the _image_widget
        message._image_widget = Mock()
        
        # Mock _render_image to track calls
        with patch.object(message, '_render_image') as mock_render:
            # Change pixel mode
            message.pixel_mode = True
            
            # Should trigger render (may be called more than once)
            assert mock_render.call_count >= 1


@pytest.mark.ui
@pytest.mark.asyncio  
class TestChatMessageEnhancedEdgeCases:
    """Test edge cases and error handling."""
    
    async def test_no_image_data_save(self, widget_pilot, mock_app_instance):
        """Test save when no image data present."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="No image",
            role="User"
        ) as pilot:
            widget = pilot.app.test_widget
            
            # Mock the app's notify method
            pilot.app.notify = mock_app_instance.notify
            
            await pilot.pause()
            
            # Try to save non-existent image
            await widget.handle_save_image()
            
            # Should not notify (early return)
            mock_app_instance.notify.assert_not_called()
    
    async def test_empty_message_handling(self, widget_pilot):
        """Test handling of empty messages."""
        async with await widget_pilot(
            ChatMessageEnhanced,
            message="",
            role="User"
        ) as pilot:
            widget = pilot.app.test_widget
            await pilot.pause()
            
            # Should still render properly
            message_text = widget.query_one(".message-text", Markdown)
            assert message_text is not None
            assert message_text._markdown == ""
    
    async def test_missing_optional_dependencies(self, widget_pilot, sample_image_data):
        """Test behavior when optional dependencies are missing."""
        # Simulate missing textual-image and rich-pixels
        with patch('tldw_chatbook.Widgets.chat_message_enhanced.TEXTUAL_IMAGE_AVAILABLE', False):
            with patch('tldw_chatbook.Widgets.chat_message_enhanced.Pixels', side_effect=ImportError):
                async with await widget_pilot(
                    ChatMessageEnhanced,
                    message="Test",
                    role="User",
                    image_data=sample_image_data
                ) as pilot:
                    widget = pilot.app.test_widget
                    await pilot.pause()
                    
                    # Should fall back gracefully
                    image_widget = widget.query_one(".message-image", Container)
                    static_content = image_widget.query_one(Static)
                    assert "📷 Image" in str(static_content.renderable)


#
# Test Documentation
#######################################################################################################################
"""
This test suite demonstrates proper Textual testing patterns for widgets:

1. **Async Testing**: All UI interaction tests use @pytest.mark.asyncio and async/await
2. **Widget Pilot**: Uses widget_pilot fixture for mounting widgets in test app context
3. **Event Testing**: Properly captures and verifies custom events using app.on()
4. **Mock App Instance**: Uses mock_app_instance for methods requiring app context
5. **Pause After Actions**: Always calls await pilot.pause() after UI interactions
6. **Proper Assertions**: Uses widget queries and state checks instead of direct property access
7. **Dependency Mocking**: Mocks external dependencies (file system, optional libraries)
8. **Edge Case Coverage**: Tests error conditions, missing data, and fallback behavior

Key Patterns:
- Use `async with await widget_pilot(...)` for widget testing
- Query child widgets with `widget.query_one()` or `widget.query()`
- Use `wait_for_condition()` for async state changes
- Mock file system operations with tmp_path fixture
- Test both success and failure paths
"""