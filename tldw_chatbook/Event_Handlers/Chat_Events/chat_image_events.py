# tldw_chatbook/Event_Handlers/Chat_Events/chat_image_events.py
# Description: Handle image operations for chat
#
# Imports
#
# Standard Library
import logging
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

# 3rd-party Libraries
from PIL import Image as PILImage

#
# Local Imports
#
#######################################################################################################################
#
# Functions:

class ChatImageHandler:
    """Handle image operations for chat."""
    
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    SUPPORTED_FORMATS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp'}
    
    @staticmethod
    async def process_image_file(file_path: str) -> Tuple[bytes, str]:
        """
        Process an image file for chat attachment.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Tuple of (image_data, mime_type)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format unsupported or too large
        """
        path = Path(file_path).expanduser().resolve()
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {file_path}")
        
        # Check file extension
        if path.suffix.lower() not in ChatImageHandler.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported image format: {path.suffix}. "
                f"Supported formats: {', '.join(ChatImageHandler.SUPPORTED_FORMATS)}"
            )
        
        # Check file size
        file_size = path.stat().st_size
        if file_size > ChatImageHandler.MAX_IMAGE_SIZE:
            raise ValueError(
                f"Image file too large ({file_size / 1024 / 1024:.1f}MB). "
                f"Maximum size: {ChatImageHandler.MAX_IMAGE_SIZE / 1024 / 1024}MB"
            )
        
        # Read image data
        image_data = path.read_bytes()
        
        # Determine MIME type
        mime_type = mimetypes.guess_type(str(path))[0] or 'image/png'
        
        # Optionally resize if too large
        try:
            processed_data = await ChatImageHandler._process_image_data(
                image_data, path.suffix.lower(), mime_type
            )
            return processed_data, mime_type
        except Exception as e:
            logging.warning(f"Failed to process image, using original: {e}")
            # If processing fails, use original data
            return image_data, mime_type
    
    @staticmethod
    async def _process_image_data(
        image_data: bytes,
        extension: str,
        mime_type: str
    ) -> bytes:
        """
        Process image data for optimization.
        
        Args:
            image_data: Raw image bytes
            extension: File extension (e.g., '.png')
            mime_type: MIME type of the image
            
        Returns:
            Processed image bytes
        """
        pil_image = PILImage.open(BytesIO(image_data))
        
        # Check if resize is needed
        max_dimension = 2048
        if pil_image.width > max_dimension or pil_image.height > max_dimension:
            # Resize maintaining aspect ratio
            pil_image.thumbnail((max_dimension, max_dimension), PILImage.Resampling.LANCZOS)
            
            # Save to bytes
            buffer = BytesIO()
            
            # Determine format
            if extension == '.png':
                format_name = 'PNG'
                save_kwargs = {'optimize': True}
            elif extension in ['.jpg', '.jpeg']:
                format_name = 'JPEG'
                save_kwargs = {'optimize': True, 'quality': 85}
            elif extension == '.webp':
                format_name = 'WEBP'
                save_kwargs = {'quality': 85}
            else:
                # For other formats, convert to PNG
                format_name = 'PNG'
                save_kwargs = {'optimize': True}
            
            pil_image.save(buffer, format=format_name, **save_kwargs)
            return buffer.getvalue()
        
        # If no resize needed, return original
        return image_data
    
    @staticmethod
    def validate_image_data(image_data: bytes) -> bool:
        """
        Validate that the bytes represent a valid image.
        
        Args:
            image_data: Image bytes to validate
            
        Returns:
            True if valid image, False otherwise
        """
        try:
            # Try to open the image
            PILImage.open(BytesIO(image_data))
            return True
        except Exception:
            return False
    
    @staticmethod
    def get_image_info(image_data: bytes) -> dict:
        """
        Get information about an image.
        
        Args:
            image_data: Image bytes
            
        Returns:
            Dictionary with image information
        """
        try:
            pil_image = PILImage.open(BytesIO(image_data))
            return {
                'width': pil_image.width,
                'height': pil_image.height,
                'format': pil_image.format,
                'mode': pil_image.mode,
                'size_kb': len(image_data) / 1024
            }
        except Exception as e:
            logging.error(f"Error getting image info: {e}")
            return {
                'error': str(e),
                'size_kb': len(image_data) / 1024
            }

#
#
#######################################################################################################################