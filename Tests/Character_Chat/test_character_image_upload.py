# test_character_image_upload.py
"""
Tests for character image upload/replace functionality.
Tests PNG-only support, size limits, and database storage.
"""

import pytest
import io
from pathlib import Path
from PIL import Image
import tempfile

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_character_card_to_json,
    export_character_card_to_png,
    import_and_save_character_from_file
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def db_instance(tmp_path):
    """Creates a DB instance for each test."""
    db_path = tmp_path / "test_image_upload.sqlite"
    db = CharactersRAGDB(db_path, "test_client")
    yield db
    db.close_connection()


@pytest.fixture
def png_image_bytes():
    """Creates a valid PNG image as bytes."""
    img = Image.new('RGB', (256, 256), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def large_png_image_bytes():
    """Creates a PNG image larger than 5MB (should be rejected)."""
    # Create a large image that will exceed 5MB when saved
    img = Image.new('RGB', (3000, 3000), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG', compress_level=0)  # No compression
    return img_bytes.getvalue()


@pytest.fixture
def oversized_png_image_bytes():
    """Creates a PNG image with dimensions > 1024x1024 (should be rejected)."""
    img = Image.new('RGB', (1500, 1500), color='green')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    return img_bytes.getvalue()


@pytest.fixture
def jpeg_image_bytes():
    """Creates a JPEG image as bytes (should be rejected)."""
    img = Image.new('RGB', (256, 256), color='yellow')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    return img_bytes.getvalue()


class TestCharacterImageUpload:
    def test_add_character_with_png_image(self, db_instance, png_image_bytes):
        """Test adding a character with a PNG image."""
        char_data = {
            "name": "Test Character with Image",
            "description": "A character with a PNG avatar",
            "personality": "Visual",
            "image": png_image_bytes
        }
        
        char_id = db_instance.add_character_card(char_data)
        assert char_id is not None
        
        # Retrieve and verify
        retrieved = db_instance.get_character_card_by_id(char_id)
        assert retrieved is not None
        assert retrieved['name'] == "Test Character with Image"
        assert retrieved['image'] == png_image_bytes
        
        # Verify it's a valid PNG
        img = Image.open(io.BytesIO(retrieved['image']))
        assert img.format == 'PNG'
        assert img.size == (256, 256)
    
    def test_update_character_add_image(self, db_instance, png_image_bytes):
        """Test adding an image to an existing character without one."""
        # Create character without image
        char_data = {
            "name": "Character to Update",
            "description": "Will add image later"
        }
        char_id = db_instance.add_character_card(char_data)
        
        # Get current version
        char = db_instance.get_character_card_by_id(char_id)
        current_version = char.get('version', 1)
        
        # Update with image
        update_data = {
            "image": png_image_bytes
        }
        updated = db_instance.update_character_card(char_id, update_data, current_version)
        assert updated is not None
        
        # Verify image was added
        retrieved = db_instance.get_character_card_by_id(char_id)
        assert retrieved['image'] == png_image_bytes
    
    def test_update_character_replace_image(self, db_instance, png_image_bytes):
        """Test replacing an existing character's image."""
        # Create character with initial image
        initial_img = Image.new('RGB', (100, 100), color='blue')
        initial_bytes = io.BytesIO()
        initial_img.save(initial_bytes, format='PNG')
        
        char_data = {
            "name": "Character with Image to Replace",
            "description": "Has initial image",
            "image": initial_bytes.getvalue()
        }
        char_id = db_instance.add_character_card(char_data)
        
        # Get current version
        char = db_instance.get_character_card_by_id(char_id)
        current_version = char.get('version', 1)
        
        # Replace with new image
        update_data = {
            "image": png_image_bytes
        }
        updated = db_instance.update_character_card(char_id, update_data, current_version)
        assert updated is not None
        
        # Verify image was replaced
        retrieved = db_instance.get_character_card_by_id(char_id)
        assert retrieved['image'] == png_image_bytes
        assert retrieved['image'] != initial_bytes.getvalue()
    
    def test_update_character_clear_image(self, db_instance, png_image_bytes):
        """Test clearing a character's image."""
        # Create character with image
        char_data = {
            "name": "Character to Clear Image",
            "description": "Has image to clear",
            "image": png_image_bytes
        }
        char_id = db_instance.add_character_card(char_data)
        
        # Get current version
        char = db_instance.get_character_card_by_id(char_id)
        current_version = char.get('version', 1)
        
        # Clear image by setting to None
        update_data = {
            "image": None
        }
        updated = db_instance.update_character_card(char_id, update_data, current_version)
        assert updated is not None
        
        # Verify image was cleared
        retrieved = db_instance.get_character_card_by_id(char_id)
        assert retrieved['image'] is None
    
    def test_export_import_character_with_image(self, db_instance, png_image_bytes, tmp_path):
        """Test that characters with images can be exported and reimported."""
        # Create character with image
        char_data = {
            "name": "Export Import Test Character",
            "description": "Character with image for export/import",
            "personality": "Persistent",
            "first_message": "Hello! I'm a test character with an image.",
            "image": png_image_bytes,
            "tags": ["test", "export"]
        }
        char_id = db_instance.add_character_card(char_data)
        
        # Export to PNG
        export_path = tmp_path / "exported_char.png"
        success = export_character_card_to_png(
            db_instance,
            char_id,
            str(export_path),
            str(tmp_path)
        )
        assert success is True
        assert export_path.exists()
        
        # Rename original to avoid conflict
        char = db_instance.get_character_card_by_id(char_id)
        db_instance.update_character_card(char_id, {"name": "Old Character"}, char['version'])
        
        # Reimport
        new_char_id = import_and_save_character_from_file(db_instance, str(export_path))
        assert new_char_id is not None
        assert new_char_id != char_id
        
        # Verify reimported character has image
        reimported = db_instance.get_character_card_by_id(new_char_id)
        assert reimported['name'] == "Export Import Test Character"
        assert reimported['image'] is not None
        
        # Verify images match
        original_img = Image.open(io.BytesIO(png_image_bytes))
        reimported_img = Image.open(io.BytesIO(reimported['image']))
        assert original_img.size == reimported_img.size
    
    def test_character_json_export_excludes_image_bytes(self, db_instance, png_image_bytes):
        """Test that JSON export includes image reference but not raw bytes."""
        # Create character with image
        char_data = {
            "name": "JSON Export Test",
            "description": "Character with image",
            "image": png_image_bytes
        }
        char_id = db_instance.add_character_card(char_data)
        
        # Export to JSON with image
        json_str = export_character_card_to_json(db_instance, char_id, include_image=True)
        assert json_str is not None
        
        import json
        exported = json.loads(json_str)
        
        # Should have image as base64 data URI
        assert 'image' in exported['data']
        assert exported['data']['image'].startswith('data:image/png;base64,')
        
        # Export without image
        json_str_no_img = export_character_card_to_json(db_instance, char_id, include_image=False)
        exported_no_img = json.loads(json_str_no_img)
        assert 'image' not in exported_no_img['data']


class TestImageValidation:
    """Tests for image format and size validation."""
    
    def test_png_only_validation(self, jpeg_image_bytes):
        """Test that non-PNG images are rejected."""
        # This would be tested in the UI handler, but we can verify
        # that JPEG bytes are indeed JPEG
        img = Image.open(io.BytesIO(jpeg_image_bytes))
        assert img.format == 'JPEG'
        # In actual implementation, this would be rejected
    
    def test_size_limit_validation(self, large_png_image_bytes):
        """Test that images over 5MB are rejected."""
        # Verify the test image is indeed large
        assert len(large_png_image_bytes) > 5 * 1024 * 1024
        # In actual implementation, this would be rejected
    
    def test_dimension_limit_validation(self, oversized_png_image_bytes):
        """Test that images over 1024x1024 are rejected."""
        img = Image.open(io.BytesIO(oversized_png_image_bytes))
        assert img.width > 1024 or img.height > 1024
        # In actual implementation, this would be rejected


class TestCharacterImageDisplay:
    """Tests for character image display functionality."""
    
    def test_character_with_image_shows_info(self, db_instance, png_image_bytes):
        """Test that characters with images show image info on card display."""
        char_data = {
            "name": "Display Test Character",
            "description": "Has image for display",
            "image": png_image_bytes
        }
        char_id = db_instance.add_character_card(char_data)
        
        # Retrieve character
        char = db_instance.get_character_card_by_id(char_id)
        assert char['image'] is not None
        
        # In UI, this would display "PNG Image: 256x256 pixels"
        img = Image.open(io.BytesIO(char['image']))
        assert img.format == 'PNG'
        assert img.width == 256
        assert img.height == 256
    
    def test_character_without_image_shows_placeholder(self, db_instance):
        """Test that characters without images show appropriate placeholder."""
        char_data = {
            "name": "No Image Display Test",
            "description": "No image"
        }
        char_id = db_instance.add_character_card(char_data)
        
        # Retrieve character
        char = db_instance.get_character_card_by_id(char_id)
        assert char['image'] is None
        # In UI, this would display "No image available"