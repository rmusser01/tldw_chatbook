# test_character_chat_lib.py
"""
Integration tests for Character Chat functionality.

These tests use real database operations with CharactersRAGDB to verify
the complete character chat system including character management,
conversations, and message handling.
"""

import pytest
import sqlite3
import json
import uuid
import base64
import io
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from PIL.PngImagePlugin import PngInfo

# Local Imports from this project
from tldw_chatbook.DB.ChaChaNotes_DB import (
    CharactersRAGDB,
    CharactersRAGDBError,
    ConflictError,
    InputError
)
from tldw_chatbook.Character_Chat.Character_Chat_Lib import (
    create_conversation,
    get_conversation_details_and_messages,
    add_message_to_conversation,
    get_character_list_for_ui,
    extract_character_id_from_ui_choice,
    load_character_and_image,
    process_db_messages_to_ui_history,
    load_chat_and_character,
    load_character_wrapper,
    import_and_save_character_from_file,
    load_chat_history_from_file_and_save_to_db,
    start_new_chat_session,
    list_character_conversations,
    update_conversation_metadata,
    post_message_to_conversation,
    retrieve_conversation_messages_for_ui
)

# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


# --- Standalone Fixtures (No conftest.py) ---

@pytest.fixture
def client_id():
    """Provides a consistent client ID for tests."""
    return "test_lib_client_001"


@pytest.fixture
def db_path(tmp_path):
    """Provides a temporary path for the database file for each test."""
    return tmp_path / "test_lib_db.sqlite"


@pytest.fixture(scope="function")
def db_instance(db_path, client_id):
    """Creates a DB instance for each test, ensuring a fresh database."""
    db = CharactersRAGDB(db_path, client_id)
    yield db
    db.close_connection()


# --- Helper Functions ---

def create_dummy_png_bytes() -> bytes:
    """Creates a 1x1 black PNG image in memory."""
    img = Image.new('RGB', (1, 1), color='black')
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG')
    return byte_arr.getvalue()


def create_dummy_png_with_chara(chara_json_str: str) -> bytes:
    """Creates a 1x1 PNG with embedded 'chara' metadata."""
    img = Image.new('RGB', (1, 1), color='red')
    # The 'chara' metadata is a base64 encoded string of the JSON
    chara_b64 = base64.b64encode(chara_json_str.encode('utf-8')).decode('utf-8')

    # Create PngInfo object for metadata
    pnginfo = PngInfo()
    # Add chara metadata as a text chunk
    pnginfo.add_text("chara", chara_b64)
    
    byte_arr = io.BytesIO()
    img.save(byte_arr, format='PNG', pnginfo=pnginfo)
    return byte_arr.getvalue()


def create_sample_v2_card_json(name: str) -> str:
    """Returns a V2 character card as a JSON string."""
    card = {
        "spec": "chara_card_v2",
        "spec_version": "2.0",
        "data": {
            "name": name,
            "description": "A test character from a V2 card.",
            "personality": "Curious",
            "scenario": "In a test.",
            "first_mes": "Hello from the V2 card!",
            "mes_example": "This is an example message.",
            "creator_notes": "",
            "system_prompt": "",
            "post_history_instructions": "",
            "tags": ["v2", "test"],
            "creator": "Tester",
            "character_version": "1.0",
            "alternate_greetings": ["Hi there!", "Greetings!"]
        }
    }
    return json.dumps(card)


# --- Test Classes ---

class TestConversationManagement:
    def test_create_conversation_with_defaults(self, db_instance: CharactersRAGDB):
        # Relies on the default character (ID 1) created by the schema
        conv_id = create_conversation(db_instance)
        assert conv_id is not None

        details = db_instance.get_conversation_by_id(conv_id)
        assert details is not None
        assert details['character_id'] == 1  # DEFAULT_CHARACTER_ID
        assert "Chat with Default Assistant" in details['title']

    def test_create_conversation_with_initial_messages(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "Talker"})
        initial_messages = [
            {'sender': 'User', 'content': 'Hi there!'},
            {'sender': 'AI', 'content': 'Hello, User!'}
        ]
        conv_id = create_conversation(db_instance, character_id=char_id, initial_messages=initial_messages)
        assert conv_id is not None

        messages = db_instance.get_messages_for_conversation(conv_id)
        assert len(messages) == 2
        assert messages[0]['sender'] == 'User'
        assert messages[1]['sender'] == 'Talker'  # Sender 'AI' is mapped to character name

    def test_get_conversation_details_and_messages(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "DetailedChar"})
        conv_id = create_conversation(db_instance, character_id=char_id, title="Test Details")
        add_message_to_conversation(db_instance, conv_id, "User", "A message")

        details = get_conversation_details_and_messages(db_instance, conv_id)
        assert details is not None
        assert details['metadata']['title'] == "Test Details"
        assert details['character_name'] == "DetailedChar"
        assert len(details['messages']) == 1
        assert details['messages'][0]['content'] == "A message"


class TestCharacterLoading:
    def test_extract_character_id_from_ui_choice(self):
        assert extract_character_id_from_ui_choice("My Character (ID: 123)") == 123
        assert extract_character_id_from_ui_choice("456") == 456
        with pytest.raises(ValueError):
            extract_character_id_from_ui_choice("Invalid Format")
        with pytest.raises(ValueError):
            extract_character_id_from_ui_choice("")

    def test_load_character_and_image(self, db_instance: CharactersRAGDB):
        image_bytes = create_dummy_png_bytes()
        card_data = {"name": "ImgChar", "first_message": "Hello, {{user}}!", "image": image_bytes}
        char_id = db_instance.add_character_card(card_data)

        char_data, history, img = load_character_and_image(db_instance, char_id, "Tester")

        assert char_data is not None
        assert char_data['name'] == "ImgChar"
        assert len(history) == 1
        assert history[0] == (None, "Hello, Tester!")  # Placeholder replaced
        assert isinstance(img, Image.Image)

    def test_process_db_messages_to_ui_history(self):
        db_messages = [
            {"sender": "User", "content": "Msg 1"},
            {"sender": "MyChar", "content": "Reply 1"},
            {"sender": "User", "content": "Msg 2a"},
            {"sender": "User", "content": "Msg 2b"},
            {"sender": "MyChar", "content": "Reply 2"},
        ]
        history = process_db_messages_to_ui_history(db_messages, "MyChar", "TestUser")
        expected = [
            ("Msg 1", "Reply 1"),
            ("Msg 2a", None),
            ("Msg 2b", "Reply 2"),
        ]
        assert history == expected

    def test_load_chat_and_character(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "Chatter", "first_message": "Hi"})
        conv_id = create_conversation(db_instance, character_id=char_id)
        add_message_to_conversation(db_instance, conv_id, "User", "Hello there")
        add_message_to_conversation(db_instance, conv_id, "Chatter", "General Kenobi")

        char_data, history, img = load_chat_and_character(db_instance, conv_id, "TestUser")

        assert char_data['name'] == "Chatter"
        assert len(history) == 1  # Only the two added messages, first_message not included in conversation history
        assert history[0] == ("Hello there", "General Kenobi")

    def test_load_character_wrapper(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "WrappedChar"})
        # Test with int ID
        char_data_int, _, _ = load_character_wrapper(db_instance, char_id, "User")
        assert char_data_int['id'] == char_id
        # Test with UI string
        char_data_str, _, _ = load_character_wrapper(db_instance, f"WrappedChar (ID: {char_id})", "User")
        assert char_data_str['id'] == char_id


class TestCharacterImport:
    def test_import_from_json_string(self, db_instance: CharactersRAGDB):
        card_name = "ImportedV2Char"
        v2_json = create_sample_v2_card_json(card_name)

        file_obj = io.BytesIO(v2_json.encode('utf-8'))
        char_id = import_and_save_character_from_file(db_instance, file_obj)

        assert char_id is not None
        retrieved = db_instance.get_character_card_by_id(char_id)
        assert retrieved['name'] == card_name
        assert retrieved['description'] == "A test character from a V2 card."

    def test_import_from_png_with_chara_metadata(self, db_instance: CharactersRAGDB):
        card_name = "PngChar"
        v2_json = create_sample_v2_card_json(card_name)
        png_bytes = create_dummy_png_with_chara(v2_json)

        file_obj = io.BytesIO(png_bytes)
        char_id = import_and_save_character_from_file(db_instance, file_obj)

        assert char_id is not None
        retrieved = db_instance.get_character_card_by_id(char_id)
        assert retrieved['name'] == card_name
        assert retrieved['image'] is not None

    def test_import_chat_history_and_save(self, db_instance: CharactersRAGDB):
        # 1. Create the character that the chat log refers to
        char_name = "LogChar"
        char_id = db_instance.add_character_card({"name": char_name})

        # 2. Create a sample chat log JSON
        chat_log = {
            "char_name": char_name,
            "history": {
                "internal": [
                    ["Hello", "Hi there"],
                    ["How are you?", "I am a test, I am fine."]
                ]
            }
        }
        chat_log_json = json.dumps(chat_log)
        file_obj = io.BytesIO(chat_log_json.encode('utf-8'))

        # 3. Import the log
        conv_id, new_char_id = load_chat_history_from_file_and_save_to_db(db_instance, file_obj)

        assert conv_id is not None
        assert new_char_id == char_id

        # 4. Verify messages were saved
        messages = db_instance.get_messages_for_conversation(conv_id)
        assert len(messages) == 4
        assert messages[0]['content'] == "Hello"
        assert messages[1]['content'] == "Hi there"
        assert messages[1]['sender'] == char_name


class TestHighLevelChatFlow:
    def test_start_new_chat_session(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "SessionStart", "first_message": "Greetings!"})

        conv_id, char_data, ui_history, img = start_new_chat_session(db_instance, char_id, "TestUser")

        assert conv_id is not None
        assert char_data['name'] == "SessionStart"
        assert ui_history == [(None, "Greetings!")]

        # Verify conversation and first message were saved to DB
        conv_details = db_instance.get_conversation_by_id(conv_id)
        assert conv_details['character_id'] == char_id

        messages = db_instance.get_messages_for_conversation(conv_id)
        assert len(messages) == 1
        assert messages[0]['content'] == "Greetings!"
        assert messages[0]['sender'] == "SessionStart"

    def test_post_message_to_conversation(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "Poster"})
        conv_id = db_instance.add_conversation({"character_id": char_id})

        # Post user message
        user_msg_id = post_message_to_conversation(db_instance, conv_id, "Poster", "User msg", is_user_message=True)
        assert user_msg_id is not None
        # Post character message
        char_msg_id = post_message_to_conversation(db_instance, conv_id, "Poster", "Char reply", is_user_message=False)
        assert char_msg_id is not None

        messages = db_instance.get_messages_for_conversation(conv_id)
        assert len(messages) == 2
        assert messages[0]['sender'] == "User"
        assert messages[1]['sender'] == "Poster"

    def test_retrieve_conversation_messages_for_ui(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "UIRetriever"})
        conv_id = start_new_chat_session(db_instance, char_id, "TestUser")[0]  # Get conv_id
        post_message_to_conversation(db_instance, conv_id, "UIRetriever", "User message", True)
        post_message_to_conversation(db_instance, conv_id, "UIRetriever", "Bot reply", False)

        ui_history = retrieve_conversation_messages_for_ui(db_instance, conv_id, "UIRetriever", "TestUser")

        # Initial message from start_new_chat_session + 1 pair
        assert len(ui_history) == 2
        assert ui_history[1] == ("User message", "Bot reply")