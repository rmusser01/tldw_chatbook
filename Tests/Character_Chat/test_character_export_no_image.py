# test_character_export_no_image.py
"""
Test character export behavior when character has no image.
"""

import pytest
import json
import tempfile
from pathlib import Path

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_character_card_to_json,
    export_character_card_to_png
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def db_instance(tmp_path):
    """Creates a DB instance for each test."""
    db_path = tmp_path / "test_export_no_image.sqlite"
    db = CharactersRAGDB(db_path, "test_client")
    yield db
    db.close_connection()


@pytest.fixture
def character_without_image(db_instance):
    """Creates a character without an image."""
    char_data = {
        "name": "No Image Character",
        "description": "A character without an avatar image",
        "personality": "Minimalist",
        "scenario": "Text-only environment",
        "first_message": "Hello! I exist without visual representation.",
        "system_prompt": "You are a character without an image.",
        "tags": ["text-only", "no-image"],
        "creator": "Test Suite",
        "character_version": "1.0"
        # Note: No 'image' field
    }
    
    char_id = db_instance.add_character_card(char_data)
    return char_id, char_data


@pytest.fixture
def character_with_image(db_instance):
    """Creates a character with an image."""
    from PIL import Image
    import io
    
    char_data = {
        "name": "Image Character",
        "description": "A character with an avatar image",
        "personality": "Visual",
        "scenario": "Graphical environment",
        "first_message": "Hello! I have a visual representation.",
        "system_prompt": "You are a character with an image.",
        "tags": ["visual", "has-image"],
        "creator": "Test Suite",
        "character_version": "1.0"
    }
    
    # Create a simple image
    img = Image.new('RGB', (100, 100), color='red')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    char_data['image'] = img_bytes.getvalue()
    
    char_id = db_instance.add_character_card(char_data)
    return char_id, char_data


def test_export_character_without_image_json_only(db_instance, character_without_image):
    """Test that characters without images export as JSON successfully."""
    char_id, original_data = character_without_image
    
    # Export as JSON
    json_str = export_character_card_to_json(db_instance, char_id, include_image=True)
    assert json_str is not None
    
    # Parse and verify JSON
    exported = json.loads(json_str)
    assert exported['spec'] == 'chara_card_v2'
    assert exported['data']['name'] == original_data['name']
    assert 'image' not in exported['data']  # No image should be in the JSON


def test_export_character_without_image_png_creates_default(db_instance, character_without_image, tmp_path):
    """Test that PNG export creates a default image when character has no image."""
    char_id, _ = character_without_image
    output_path = tmp_path / "no_image_char.png"
    
    # This should still succeed by creating a default gray image
    success = export_character_card_to_png(
        db_instance,
        char_id,
        str(output_path),
        str(tmp_path)
    )
    assert success is True
    assert output_path.exists()
    
    # Verify it's a valid PNG
    from PIL import Image
    img = Image.open(output_path)
    assert img.format == 'PNG'
    assert img.size == (256, 256)  # Default size


def test_export_character_with_image_both_formats(db_instance, character_with_image, tmp_path):
    """Test that characters with images can export as both JSON and PNG."""
    char_id, original_data = character_with_image
    
    # Export as JSON
    json_str = export_character_card_to_json(db_instance, char_id, include_image=True)
    assert json_str is not None
    
    exported = json.loads(json_str)
    assert 'image' in exported['data']  # Image should be included as base64
    
    # Export as PNG
    png_path = tmp_path / "with_image_char.png"
    success = export_character_card_to_png(
        db_instance,
        char_id,
        str(png_path),
        str(tmp_path)
    )
    assert success is True
    assert png_path.exists()


def test_event_handler_logic_simulation():
    """Simulate the event handler logic for characters with and without images."""
    # Simulate character data with image
    char_with_image = {
        'id': 1,
        'name': 'ImageChar',
        'image': b'fake_image_bytes'
    }
    
    # Simulate character data without image
    char_without_image = {
        'id': 2,
        'name': 'NoImageChar',
        # No 'image' key
    }
    
    # Test the condition used in the event handler
    assert char_with_image.get('image') is not None
    assert char_without_image.get('image') is None
    
    # This matches the logic in the event handler
    if char_with_image.get('image'):
        export_type = "JSON and PNG"
    else:
        export_type = "JSON only"
    assert export_type == "JSON and PNG"
    
    if char_without_image.get('image'):
        export_type = "JSON and PNG"
    else:
        export_type = "JSON only"
    assert export_type == "JSON only"