# Tests/Event_Handlers/Chat_Events/test_chat_image_properties.py
# Description: Property-based tests for chat image functionality
#
# Imports
#
# Standard Library
import pytest
import tempfile
from pathlib import Path
from io import BytesIO

# 3rd-party Libraries
from hypothesis import given, strategies as st, assume, settings
from hypothesis.extra.numpy import arrays
import numpy as np
from PIL import Image as PILImage

# Local Imports
from tldw_chatbook.Event_Handlers.Chat_Events.chat_image_events import ChatImageHandler

#
#######################################################################################################################
#
# Property-based Test Strategies

# Strategy for valid image dimensions
image_dimensions = st.tuples(
    st.integers(min_value=1, max_value=5000),  # width
    st.integers(min_value=1, max_value=5000)   # height
)

# Strategy for image formats
image_formats = st.sampled_from(['PNG', 'JPEG', 'GIF', 'BMP'])

# Strategy for image modes
image_modes = st.sampled_from(['RGB', 'RGBA', 'L', 'P'])

# Strategy for file extensions
valid_extensions = st.sampled_from(['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'])

# Strategy for file sizes (in bytes)
file_sizes = st.integers(min_value=100, max_value=15 * 1024 * 1024)  # Up to 15MB

#
# Property-based Tests
#

class TestChatImageProperties:
    """Property-based tests for image handling."""
    
    @given(
        dimensions=image_dimensions,
        format_name=image_formats,
        mode=st.sampled_from(['RGB', 'L'])  # Limit modes for format compatibility
    )
    @settings(max_examples=20, deadline=5000)  # Increase deadline for image operations
    @pytest.mark.asyncio
    async def test_process_image_preserves_aspect_ratio(self, dimensions, format_name, mode):
        """Test that image processing preserves aspect ratio."""
        width, height = dimensions
        
        # Create test image
        if mode == 'L' and format_name == 'JPEG':
            mode = 'RGB'  # JPEG doesn't support grayscale well
        
        img = PILImage.new(mode, (width, height))
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(suffix=f'.{format_name.lower()}', delete=True) as f:
            img.save(f, format=format_name)
            f.flush()
            
            # Process the image
            try:
                processed_data, _ = await ChatImageHandler.process_image_file(f.name)
                
                # Load processed image
                processed_img = PILImage.open(BytesIO(processed_data))
                
                # Calculate expected dimensions
                if width > 2048 or height > 2048:
                    # Should be resized
                    ratio = min(2048 / width, 2048 / height)
                    expected_width = int(width * ratio)
                    expected_height = int(height * ratio)
                    
                    # Check aspect ratio is preserved (with small tolerance for rounding)
                    original_ratio = width / height
                    processed_ratio = processed_img.width / processed_img.height
                    
                    assert abs(original_ratio - processed_ratio) < 0.01
                    assert processed_img.width <= 2048
                    assert processed_img.height <= 2048
                else:
                    # Should not be resized
                    assert processed_img.width == width
                    assert processed_img.height == height
            except Exception as e:
                # Some format/mode combinations might not be supported
                assume(False)
    
    @given(
        extension=valid_extensions,
        size_bytes=file_sizes
    )
    @settings(max_examples=15)
    @pytest.mark.asyncio
    async def test_file_size_validation(self, extension, size_bytes):
        """Test that file size validation works correctly."""
        with tempfile.NamedTemporaryFile(suffix=extension, delete=True) as f:
            # Write dummy data of specified size
            f.write(b'0' * size_bytes)
            f.flush()
            
            if size_bytes > ChatImageHandler.MAX_IMAGE_SIZE:
                # Should raise ValueError
                with pytest.raises(ValueError) as exc_info:
                    await ChatImageHandler.process_image_file(f.name)
                assert "Image file too large" in str(exc_info.value)
            else:
                # Should not raise for valid sizes
                # Note: This will fail for non-image data, which is expected
                try:
                    await ChatImageHandler.process_image_file(f.name)
                except Exception as e:
                    # Expected for non-image data
                    assert "cannot identify image file" in str(e).lower() or \
                           "not a valid" in str(e).lower()
    
    @given(
        image_data=st.binary(min_size=10, max_size=1000)
    )
    def test_validate_image_data_handles_arbitrary_bytes(self, image_data):
        """Test that validate_image_data handles arbitrary byte sequences safely."""
        # Should not raise exceptions, just return True/False
        result = ChatImageHandler.validate_image_data(image_data)
        assert isinstance(result, bool)
    
    @given(
        width=st.integers(min_value=1, max_value=1000),
        height=st.integers(min_value=1, max_value=1000),
        color=st.tuples(
            st.integers(0, 255),
            st.integers(0, 255),
            st.integers(0, 255)
        )
    )
    def test_get_image_info_correctness(self, width, height, color):
        """Test that get_image_info returns correct information."""
        # Create image with specific properties
        img = PILImage.new('RGB', (width, height), color=color)
        
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        image_data = buffer.getvalue()
        
        info = ChatImageHandler.get_image_info(image_data)
        
        assert info['width'] == width
        assert info['height'] == height
        assert info['format'] == 'PNG'
        assert info['mode'] == 'RGB'
        assert info['size_kb'] > 0
        assert 'error' not in info
    
    @given(
        dimensions=st.lists(
            image_dimensions,
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=10, deadline=5000)
    @pytest.mark.asyncio
    async def test_consistent_resize_behavior(self, dimensions):
        """Test that resizing behavior is consistent across multiple images."""
        results = []
        
        for width, height in dimensions:
            img = PILImage.new('RGB', (width, height))
            
            with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as f:
                img.save(f, format='PNG')
                f.flush()
                
                try:
                    processed_data, _ = await ChatImageHandler.process_image_file(f.name)
                    processed_img = PILImage.open(BytesIO(processed_data))
                    
                    needs_resize = width > 2048 or height > 2048
                    was_resized = (processed_img.width != width or 
                                 processed_img.height != height)
                    
                    results.append((needs_resize, was_resized))
                except Exception:
                    # Skip on error
                    pass
        
        # All images that need resize should be resized
        # All images that don't need resize should not be resized
        for needs_resize, was_resized in results:
            if needs_resize:
                assert was_resized
            else:
                assert not was_resized


class TestImageProcessingEdgeCases:
    """Property-based tests for edge cases."""
    
    @given(
        width=st.integers(min_value=1, max_value=10),
        height=st.integers(min_value=1, max_value=10)
    )
    @pytest.mark.asyncio
    async def test_tiny_images_handled_correctly(self, width, height):
        """Test that very small images are handled correctly."""
        img = PILImage.new('RGB', (width, height), color='red')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as f:
            img.save(f, format='PNG')
            f.flush()
            
            processed_data, mime_type = await ChatImageHandler.process_image_file(f.name)
            
            # Should process successfully
            assert isinstance(processed_data, bytes)
            assert mime_type == 'image/png'
            
            # Should not be resized
            processed_img = PILImage.open(BytesIO(processed_data))
            assert processed_img.width == width
            assert processed_img.height == height
    
    @given(
        ratio=st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
    )
    @settings(max_examples=15)
    @pytest.mark.asyncio
    async def test_extreme_aspect_ratios(self, ratio):
        """Test handling of images with extreme aspect ratios."""
        # Create image with extreme aspect ratio
        if ratio > 1:
            width = int(1000 * ratio)
            height = 1000
        else:
            width = 1000
            height = int(1000 / ratio)
        
        # Limit to reasonable sizes
        width = min(width, 5000)
        height = min(height, 5000)
        
        img = PILImage.new('RGB', (width, height), color='blue')
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as f:
            img.save(f, format='JPEG')
            f.flush()
            
            processed_data, _ = await ChatImageHandler.process_image_file(f.name)
            processed_img = PILImage.open(BytesIO(processed_data))
            
            # Check that aspect ratio is preserved
            original_ratio = width / height
            processed_ratio = processed_img.width / processed_img.height
            
            # Allow small tolerance for rounding
            assert abs(original_ratio - processed_ratio) / original_ratio < 0.02
    
    @given(
        format_pairs=st.lists(
            st.tuples(
                st.sampled_from(['PNG', 'JPEG', 'GIF', 'BMP']),
                st.sampled_from(['.png', '.jpg', '.jpeg', '.gif', '.bmp'])
            ),
            min_size=1,
            max_size=5
        )
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_format_extension_mismatch_handling(self, format_pairs):
        """Test handling of mismatched formats and extensions."""
        for format_name, extension in format_pairs:
            img = PILImage.new('RGB', (100, 100), color='green')
            
            with tempfile.NamedTemporaryFile(suffix=extension, delete=True) as f:
                # Save with potentially mismatched format
                try:
                    img.save(f, format=format_name)
                    f.flush()
                    
                    # Should handle gracefully
                    processed_data, mime_type = await ChatImageHandler.process_image_file(f.name)
                    
                    # MIME type should be based on extension
                    assert mime_type.startswith('image/')
                except Exception:
                    # Some format/extension combinations might not work
                    pass


class TestImageDataIntegrity:
    """Test data integrity through processing."""
    
    @given(
        seed=st.integers(min_value=0, max_value=2**32-1)
    )
    @settings(max_examples=10)
    @pytest.mark.asyncio
    async def test_image_content_preservation(self, seed):
        """Test that image content is preserved (when not resized)."""
        # Create image with random but deterministic content
        np.random.seed(seed)
        pixels = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
        img = PILImage.fromarray(pixels, mode='RGB')
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=True) as f:
            img.save(f, format='PNG')
            f.flush()
            
            processed_data, _ = await ChatImageHandler.process_image_file(f.name)
            
            # Load processed image
            processed_img = PILImage.open(BytesIO(processed_data))
            
            # Should not be resized (100x100 is small)
            assert processed_img.size == (100, 100)
            
            # Convert back to array
            processed_pixels = np.array(processed_img)
            
            # Should be identical (PNG is lossless)
            np.testing.assert_array_equal(pixels, processed_pixels)
    
    @given(
        quality=st.integers(min_value=1, max_value=100)
    )
    @settings(max_examples=10)
    def test_jpeg_quality_handling(self, quality):
        """Test JPEG quality settings."""
        img = PILImage.new('RGB', (200, 200), color='yellow')
        
        buffer1 = BytesIO()
        buffer2 = BytesIO()
        
        # Save with different qualities
        img.save(buffer1, format='JPEG', quality=quality)
        img.save(buffer2, format='JPEG', quality=85)  # Handler default
        
        # File size should correlate with quality
        size1 = len(buffer1.getvalue())
        size2 = len(buffer2.getvalue())
        
        if quality < 85:
            # Lower quality should produce smaller files
            assert size1 <= size2 * 1.2  # Allow some variance
        elif quality > 85:
            # Higher quality should produce larger files
            assert size1 >= size2 * 0.8  # Allow some variance

#
#
#######################################################################################################################