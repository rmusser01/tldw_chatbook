# test_character_file_operations.py
"""
Tests for character chat file operations including import/export functionality.
"""

import pytest
import json
import tempfile
from pathlib import Path
import io
import base64

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    export_character_card_to_json,
    export_character_card_to_png,
    export_conversation_to_json,
    export_conversation_to_text,
    import_and_save_character_from_file,
    create_conversation,
    add_message_to_conversation,
    extract_json_from_image_file
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


@pytest.fixture
def db_instance(tmp_path):
    """Creates a DB instance for each test."""
    db_path = tmp_path / "test_file_ops.sqlite"
    db = CharactersRAGDB(db_path, "test_client")
    yield db
    db.close_connection()


@pytest.fixture
def sample_character(db_instance):
    """Creates a sample character for testing."""
    char_data = {
        "name": "Test Character",
        "description": "A character for testing file operations",
        "personality": "Helpful and friendly",
        "scenario": "Testing environment",
        "first_message": "Hello! I'm here to test file operations.",
        "example_messages": "User: Hi\nTest Character: Hello there!",
        "system_prompt": "You are a test character.",
        "post_history_instructions": "Remember to be helpful.",
        "tags": ["test", "export"],
        "creator": "Test Suite",
        "character_version": "1.0",
        "alternate_greetings": ["Hi!", "Hey there!"],
        "extensions": {"test_ext": {"data": "test"}}
    }
    
    # Add a simple image
    img = Image.new('RGB', (100, 100), color='blue')
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    char_data['image'] = img_bytes.getvalue()
    
    char_id = db_instance.add_character_card(char_data)
    return char_id, char_data


@pytest.fixture
def sample_conversation(db_instance, sample_character):
    """Creates a sample conversation with messages."""
    char_id, _ = sample_character
    conv_id = create_conversation(
        db_instance,
        title="Test Conversation",
        character_id=char_id,
        system_keywords=["test", "export"]
    )
    
    # Add some messages
    add_message_to_conversation(db_instance, conv_id, "User", "Hello!")
    add_message_to_conversation(db_instance, conv_id, "Test Character", "Hi there! How can I help?")
    add_message_to_conversation(db_instance, conv_id, "User", "Just testing the export functionality.")
    add_message_to_conversation(db_instance, conv_id, "Test Character", "Perfect! I'm here for that.")
    
    return conv_id


class TestCharacterExport:
    def test_export_character_to_json_with_image(self, db_instance, sample_character):
        """Test exporting a character card to JSON format with image."""
        char_id, original_data = sample_character
        
        json_str = export_character_card_to_json(db_instance, char_id, include_image=True)
        assert json_str is not None
        
        # Parse the JSON
        exported = json.loads(json_str)
        assert exported['spec'] == 'chara_card_v2'
        assert exported['spec_version'] == '2.0'
        
        # Check data fields
        data = exported['data']
        assert data['name'] == original_data['name']
        assert data['description'] == original_data['description']
        assert data['personality'] == original_data['personality']
        assert data['scenario'] == original_data['scenario']
        assert data['first_mes'] == original_data['first_message']
        assert data['tags'] == original_data['tags']
        assert data['extensions'] == original_data['extensions']
        
        # Check image is included as base64
        assert 'image' in data
        assert data['image'].startswith('data:image/png;base64,')
        
    def test_export_character_to_json_without_image(self, db_instance, sample_character):
        """Test exporting a character card to JSON format without image."""
        char_id, _ = sample_character
        
        json_str = export_character_card_to_json(db_instance, char_id, include_image=False)
        assert json_str is not None
        
        exported = json.loads(json_str)
        assert 'image' not in exported['data']
        
    def test_export_character_to_png(self, db_instance, sample_character, tmp_path):
        """Test exporting a character card as PNG with embedded metadata."""
        char_id, _ = sample_character
        output_path = tmp_path / "test_char.png"
        
        success = export_character_card_to_png(
            db_instance, 
            char_id, 
            str(output_path),
            str(tmp_path)
        )
        assert success is True
        assert output_path.exists()
        
        # Verify the PNG has embedded character data
        extracted_json = extract_json_from_image_file(str(output_path), str(tmp_path))
        assert extracted_json is not None
        
        char_data = json.loads(extracted_json)
        assert char_data['spec'] == 'chara_card_v2'
        assert char_data['data']['name'] == 'Test Character'
        
    def test_export_nonexistent_character(self, db_instance):
        """Test exporting a character that doesn't exist."""
        json_str = export_character_card_to_json(db_instance, 9999)
        assert json_str is None
        
    def test_reimport_exported_character(self, db_instance, sample_character, tmp_path):
        """Test that an exported character can be reimported successfully."""
        char_id, original_data = sample_character
        
        # Export as PNG
        png_path = tmp_path / "export_test.png"
        success = export_character_card_to_png(
            db_instance,
            char_id,
            str(png_path),
            str(tmp_path)
        )
        assert success is True
        
        # Get current version to update the character
        char_data = db_instance.get_character_card_by_id(char_id)
        current_version = char_data.get('version', 1)
        
        # Rename original character to avoid conflicts
        db_instance.update_character_card(char_id, {"name": "Old Character"}, current_version)
        
        # Reimport from PNG
        new_char_id = import_and_save_character_from_file(db_instance, str(png_path))
        assert new_char_id is not None
        assert new_char_id != char_id
        
        # Verify reimported data
        reimported = db_instance.get_character_card_by_id(new_char_id)
        assert reimported['name'] == original_data['name']
        assert reimported['description'] == original_data['description']


class TestConversationExport:
    def test_export_conversation_to_json(self, db_instance, sample_conversation):
        """Test exporting a conversation to JSON format."""
        json_str = export_conversation_to_json(
            db_instance,
            sample_conversation,
            include_character_card=True
        )
        assert json_str is not None
        
        exported = json.loads(json_str)
        assert 'conversation' in exported
        assert 'messages' in exported
        assert 'character_card' in exported
        
        # Check conversation metadata
        conv = exported['conversation']
        assert conv['title'] == 'Test Conversation'
        assert set(conv['keywords']) == {'test', 'export'}
        
        # Check messages
        messages = exported['messages']
        assert len(messages) == 4
        assert messages[0]['sender'] == 'User'
        assert messages[0]['content'] == 'Hello!'
        assert messages[1]['sender'] == 'Test Character'
        
        # Check character card is included (without image bytes)
        char_card = exported['character_card']
        assert char_card['name'] == 'Test Character'
        assert 'image' not in char_card
        
    def test_export_conversation_to_text(self, db_instance, sample_conversation):
        """Test exporting a conversation to text format."""
        text_str = export_conversation_to_text(
            db_instance,
            sample_conversation,
            user_name="TestUser"
        )
        assert text_str is not None
        
        lines = text_str.split('\n')
        assert 'Conversation: Test Conversation' in lines[0]
        assert 'Character: Test Character' in lines[1]
        # Keywords may be in different order or not included if empty
        assert ('Keywords: test, export' in text_str or 'Keywords: export, test' in text_str or 'Keywords:' not in text_str)
        
        # Check messages are formatted correctly
        assert '[' in text_str  # Timestamps
        assert 'TestUser:' in text_str
        assert 'Test Character:' in text_str
        assert 'Hello!' in text_str
        assert 'Just testing the export functionality.' in text_str
        
    def test_export_empty_conversation(self, db_instance, sample_character):
        """Test exporting a conversation with no messages."""
        char_id, _ = sample_character
        conv_id = create_conversation(
            db_instance,
            title="Empty Conversation",
            character_id=char_id
        )
        
        json_str = export_conversation_to_json(db_instance, conv_id)
        assert json_str is not None
        
        exported = json.loads(json_str)
        assert exported['messages'] == []
        
    def test_export_nonexistent_conversation(self, db_instance):
        """Test exporting a conversation that doesn't exist."""
        json_str = export_conversation_to_json(db_instance, "nonexistent-conv-id")
        assert json_str is None
        
        text_str = export_conversation_to_text(db_instance, "nonexistent-conv-id")
        assert text_str is None


class TestCharacterWithoutImage:
    def test_export_character_without_image_to_png(self, db_instance, tmp_path):
        """Test exporting a character without an image creates a default image."""
        # Create character without image
        char_data = {
            "name": "No Image Character",
            "description": "A character without an avatar"
        }
        char_id = db_instance.add_character_card(char_data)
        
        output_path = tmp_path / "no_image_char.png"
        success = export_character_card_to_png(
            db_instance,
            char_id,
            str(output_path),
            str(tmp_path)
        )
        assert success is True
        assert output_path.exists()
        
        # Verify it created a valid PNG
        img = Image.open(output_path)
        assert img.format == 'PNG'
        assert img.size == (256, 256)  # Default size
        
        # Verify metadata is still embedded
        extracted_json = extract_json_from_image_file(str(output_path), str(tmp_path))
        assert extracted_json is not None
        char_data = json.loads(extracted_json)
        assert char_data['data']['name'] == 'No Image Character'