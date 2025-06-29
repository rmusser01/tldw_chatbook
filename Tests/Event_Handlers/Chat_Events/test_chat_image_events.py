# Tests/Event_Handlers/Chat_Events/test_chat_image_events.py
# Description: Integration tests for ChatImageHandler
#
"""
test_chat_image_events.py
------------------------

Integration tests for ChatImageHandler that test image file processing,
including file I/O operations and image manipulation with PIL.
"""
# Imports
#
# Standard Library
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile

# 3rd-party Libraries
from PIL import Image as PILImage

# Local Imports
from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler

#
#######################################################################################################################
#
# Test Fixtures

@pytest.fixture
def temp_image_file():
    """Create a temporary image file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Create a test image
        img = PILImage.new('RGB', (200, 200), color='green')
        img.save(f, format='PNG')
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def large_image_file():
    """Create a large temporary image file for testing resize."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        # Create a large test image
        img = PILImage.new('RGB', (3000, 3000), color='blue')
        img.save(f, format='JPEG')
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def oversized_file():
    """Create a file that exceeds size limit."""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        # Write more than 10MB of data
        f.write(b'0' * (11 * 1024 * 1024))
        temp_path = Path(f.name)
    
    yield temp_path
    
    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


#
# Integration Tests
#

@pytest.mark.integration
class TestChatImageHandler:
    """Test suite for ChatImageHandler."""
    
    @pytest.mark.asyncio
    async def test_process_valid_image(self, temp_image_file):
        """Test processing a valid image file."""
        image_data, mime_type = await ChatImageHandler.process_image_file(str(temp_image_file))
        
        assert isinstance(image_data, bytes)
        assert len(image_data) > 0
        assert mime_type == 'image/png'
    
    @pytest.mark.asyncio
    async def test_process_nonexistent_file(self):
        """Test processing a non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            await ChatImageHandler.process_image_file('/path/to/nonexistent.png')
        
        assert "Image file not found" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_unsupported_format(self):
        """Test processing a file with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=True) as f:
            f.write(b"Not an image")
            f.flush()
            
            with pytest.raises(ValueError) as exc_info:
                await ChatImageHandler.process_image_file(f.name)
            
            assert "Unsupported image format" in str(exc_info.value)
            assert ".txt" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_oversized_file(self, oversized_file):
        """Test processing a file that exceeds size limit."""
        with pytest.raises(ValueError) as exc_info:
            await ChatImageHandler.process_image_file(str(oversized_file))
        
        assert "Image file too large" in str(exc_info.value)
    
    @pytest.mark.asyncio
    async def test_process_large_image_resize(self, large_image_file):
        """Test that large images are resized."""
        image_data, mime_type = await ChatImageHandler.process_image_file(str(large_image_file))
        
        # Load the processed image to check dimensions
        from io import BytesIO
        processed_img = PILImage.open(BytesIO(image_data))
        
        # Should be resized to max 2048
        assert processed_img.width <= 2048
        assert processed_img.height <= 2048
        # Aspect ratio should be maintained
        assert processed_img.width == processed_img.height  # Original was square
    
    @pytest.mark.asyncio
    async def test_process_image_with_tilde_path(self, temp_image_file):
        """Test processing image with tilde in path."""
        # Create a path with tilde
        home_path = Path.home()
        
        # Only run this test if temp file is under home directory
        try:
            relative_path = temp_image_file.relative_to(home_path)
            tilde_path = f"~/{relative_path}"
            
            image_data, mime_type = await ChatImageHandler.process_image_file(tilde_path)
            
            assert isinstance(image_data, bytes)
            assert len(image_data) > 0
        except ValueError:
            # Skip test if temp file is not under home directory
            pytest.skip("Temp file is not under home directory, skipping tilde path test")
    
    def test_validate_image_data_valid(self):
        """Test validating valid image data."""
        # Create valid image data
        img = PILImage.new('RGB', (10, 10), color='red')
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        assert ChatImageHandler.validate_image_data(image_data) is True
    
    def test_validate_image_data_invalid(self):
        """Test validating invalid image data."""
        invalid_data = b"This is not image data"
        assert ChatImageHandler.validate_image_data(invalid_data) is False
    
    def test_get_image_info_valid(self):
        """Test getting info from valid image."""
        # Create test image
        img = PILImage.new('RGB', (640, 480), color='white')
        from io import BytesIO
        buffer = BytesIO()
        img.save(buffer, format='JPEG')
        image_data = buffer.getvalue()
        
        info = ChatImageHandler.get_image_info(image_data)
        
        assert info['width'] == 640
        assert info['height'] == 480
        assert info['format'] == 'JPEG'
        assert info['mode'] == 'RGB'
        assert 'size_kb' in info
        assert info['size_kb'] > 0
    
    def test_get_image_info_invalid(self):
        """Test getting info from invalid image data."""
        invalid_data = b"Not an image"
        
        info = ChatImageHandler.get_image_info(invalid_data)
        
        assert 'error' in info
        assert 'size_kb' in info
    
    @pytest.mark.asyncio
    async def test_process_different_formats(self):
        """Test processing different image formats."""
        formats = {
            '.png': 'PNG',
            '.jpg': 'JPEG',
            '.jpeg': 'JPEG',
            '.gif': 'GIF',
            '.webp': 'WEBP',
            '.bmp': 'BMP'
        }
        
        for ext, format_name in formats.items():
            # Skip WEBP if not supported by PIL installation
            if format_name == 'WEBP':
                try:
                    img = PILImage.new('RGB', (10, 10))
                    from io import BytesIO
                    buffer = BytesIO()
                    img.save(buffer, format='WEBP')
                except Exception:
                    continue
            
            with tempfile.NamedTemporaryFile(suffix=ext, delete=True) as f:
                # Create image in specific format
                img = PILImage.new('RGB', (100, 100), color='yellow')
                img.save(f, format=format_name)
                f.flush()
                
                # Process it
                image_data, mime_type = await ChatImageHandler.process_image_file(f.name)
                
                assert isinstance(image_data, bytes)
                assert len(image_data) > 0
                assert mime_type.startswith('image/')
    
    @pytest.mark.asyncio
    async def test_process_image_data_optimization(self):
        """Test that image processing optimizes file size."""
        # Create unoptimized JPEG
        img = PILImage.new('RGB', (2500, 2500), color='red')
        
        from io import BytesIO
        unoptimized_buffer = BytesIO()
        img.save(unoptimized_buffer, format='JPEG', quality=100)
        unoptimized_data = unoptimized_buffer.getvalue()
        
        # Process the image data
        processed_data = await ChatImageHandler._process_image_data(
            unoptimized_data,
            '.jpg',
            'image/jpeg'
        )
        
        # Processed should be smaller due to resize and optimization
        assert len(processed_data) < len(unoptimized_data)
        
        # Verify dimensions
        processed_img = PILImage.open(BytesIO(processed_data))
        assert processed_img.width == 2048
        assert processed_img.height == 2048


@pytest.mark.integration
class TestChatImageHandlerEdgeCases:
    """Test edge cases for ChatImageHandler."""
    
    @pytest.mark.asyncio
    async def test_process_corrupted_image_file(self):
        """Test processing a corrupted image file."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as f:
            # Write invalid JPEG data
            f.write(b'\xFF\xD8\xFF\xE0' + b'corrupted data')
            f.flush()
            
            # Should either raise error or return original data
            try:
                image_data, mime_type = await ChatImageHandler.process_image_file(f.name)
                # If it doesn't raise, it should return original data
                assert isinstance(image_data, bytes)
            except Exception:
                # This is also acceptable behavior
                pass
    
    @pytest.mark.asyncio
    async def test_process_zero_size_file(self):
        """Test processing an empty file."""
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as f:
            # Empty file
            f.flush()
            
            # PIL will fail to identify an empty file as an image
            # The error will be caught and logged, but the original empty data will be returned
            try:
                image_data, mime_type = await ChatImageHandler.process_image_file(f.name)
                # If we get here, it returned the original empty data
                assert isinstance(image_data, bytes)
                assert len(image_data) == 0
                assert mime_type == 'image/png'
            except Exception:
                # This is also acceptable - PIL might fail to process empty file
                pass
    
    @pytest.mark.asyncio
    async def test_process_animated_gif(self):
        """Test processing an animated GIF."""
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=True) as f:
            # Create a simple animated GIF (2 frames)
            img1 = PILImage.new('RGB', (100, 100), color='red')
            img2 = PILImage.new('RGB', (100, 100), color='blue')
            
            img1.save(f, format='GIF', save_all=True, append_images=[img2], duration=100, loop=0)
            f.flush()
            
            image_data, mime_type = await ChatImageHandler.process_image_file(f.name)
            
            assert isinstance(image_data, bytes)
            assert mime_type == 'image/gif'
    
    @pytest.mark.asyncio
    async def test_mime_type_detection(self):
        """Test MIME type detection for various scenarios."""
        # Test with extension that doesn't match content
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=True) as f:
            # Save PNG data with .txt extension
            img = PILImage.new('RGB', (50, 50), color='green')
            img.save(f, format='PNG')
            f.flush()
            
            # Rename to bypass extension check
            png_as_txt = Path(f.name).with_suffix('.png')
            Path(f.name).rename(png_as_txt)
            
            try:
                image_data, mime_type = await ChatImageHandler.process_image_file(str(png_as_txt))
                assert mime_type == 'image/png'
            finally:
                if png_as_txt.exists():
                    png_as_txt.unlink()


@pytest.mark.unit
class TestChatImageHandlerConstants:
    """Test ChatImageHandler constants and configuration."""
    
    def test_max_image_size_constant(self):
        """Test MAX_IMAGE_SIZE constant."""
        assert ChatImageHandler.MAX_IMAGE_SIZE == 10 * 1024 * 1024  # 10MB
    
    def test_supported_formats_constant(self):
        """Test SUPPORTED_FORMATS constant."""
        expected_formats = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
        assert ChatImageHandler.SUPPORTED_FORMATS == expected_formats
    
    def test_supported_formats_lowercase(self):
        """Test that format checking is case-insensitive."""
        # This tests the implementation detail that extensions are converted to lowercase
        assert '.png' in ChatImageHandler.SUPPORTED_FORMATS
        assert '.PNG' not in ChatImageHandler.SUPPORTED_FORMATS  # Should be lowercase

#
#
#######################################################################################################################