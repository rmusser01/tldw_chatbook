# test_chat_functions.py
#
# Imports
import pytest
import base64
import io
from unittest.mock import patch, MagicMock
#
# 3rd-party Libraries
import requests
from PIL import Image
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError
from tldw_chatbook.Chat.Chat_Functions import (
    chat_api_call,
    chat,
    save_chat_history_to_db_wrapper,
    save_character,
    load_characters,
    get_character_names,
    parse_user_dict_markdown_file,
    process_user_input,
    ChatDictionary,
    DEFAULT_CHARACTER_NAME
)
from tldw_chatbook.Chat.Chat_Deps import (
    ChatBadRequestError,
    ChatAuthenticationError,
    ChatRateLimitError,
    ChatProviderError,
    ChatAPIError
)
#
#######################################################################################################################
#
# --- Standalone Fixtures (No conftest.py) ---

@pytest.fixture
def client_id():
    """Provides a consistent client ID for tests."""
    return "test_chat_func_client"


@pytest.fixture
def db_path(tmp_path):
    """Provides a temporary path for the database file for each test."""
    return tmp_path / "test_chat_func_db.sqlite"


@pytest.fixture(scope="function")
def db_instance(db_path, client_id):
    """Creates a DB instance for each test, ensuring a fresh database."""
    db = CharactersRAGDB(db_path, client_id)
    yield db
    db.close_connection()


# --- Helper Functions ---

def create_base64_image():
    """Creates a dummy 1x1 png and returns its base64 string."""
    img_bytes = io.BytesIO()
    Image.new('RGB', (1, 1)).save(img_bytes, format='PNG')
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')


# --- Test Classes ---

@patch('tldw_chatbook.Chat.Chat_Functions.API_CALL_HANDLERS')
@pytest.mark.unit
class TestChatApiCall:
    def test_routes_to_correct_handler(self, mock_handlers, mocker):
        mock_openai_handler = mocker.MagicMock(return_value="OpenAI response")
        mock_openai_handler.__name__ = "mock_openai_handler"
        mock_handlers.get.return_value = mock_openai_handler

        response = chat_api_call(
            api_endpoint="openai",
            messages_payload=[{"role": "user", "content": "test"}],
            model="gpt-4"
        )

        mock_handlers.get.assert_called_with("openai")
        mock_openai_handler.assert_called_once()
        kwargs = mock_openai_handler.call_args.kwargs
        assert kwargs['input_data'][0]['content'] == "test"  # Mapped to 'input_data' for openai
        assert kwargs['model'] == "gpt-4"
        assert response == "OpenAI response"

    def test_unsupported_endpoint_raises_error(self, mock_handlers):
        mock_handlers.get.return_value = None
        with pytest.raises(ValueError, match="Unsupported API endpoint: unsupported"):
            chat_api_call("unsupported", messages_payload=[])

    def test_http_error_401_raises_auth_error(self, mock_handlers, mocker):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.text = "Invalid API key"
        http_error = requests.exceptions.HTTPError(response=mock_response)

        mock_handler = mocker.MagicMock(side_effect=http_error)
        mock_handler.__name__ = "mock_handler"
        mock_handlers.get.return_value = mock_handler

        with pytest.raises(ChatAuthenticationError):
            chat_api_call("openai", messages_payload=[])


@pytest.mark.unit
class TestChatFunction:
    @patch('tldw_chatbook.Chat.Chat_Functions.chat_api_call')
    def test_chat_basic_flow(self, mock_chat_api_call):
        mock_chat_api_call.return_value = "LLM says hi"

        response = chat(
            message="Hello",
            history=[],
            media_content=None,
            selected_parts=[],
            api_endpoint="openai",
            api_key="sk-123",
            model="gpt-4",
            temperature=0.7,
            custom_prompt="Be brief."
        )

        assert response == "LLM says hi"
        mock_chat_api_call.assert_called_once()
        kwargs = mock_chat_api_call.call_args.kwargs

        assert kwargs['api_endpoint'] == 'openai'
        assert kwargs['model'] == 'gpt-4'
        payload = kwargs['messages_payload']
        assert len(payload) == 1
        assert payload[0]['role'] == 'user'
        user_content = payload[0]['content']
        assert isinstance(user_content, list)
        assert user_content[0]['type'] == 'text'
        assert user_content[0]['text'] == "Be brief.\n\nHello"

    @patch('tldw_chatbook.Chat.Chat_Functions.chat_api_call')
    def test_chat_with_image_and_rag(self, mock_chat_api_call):
        b64_img = create_base64_image()

        chat(
            message="Describe this.",
            history=[],
            media_content={"summary": "This is a summary."},
            selected_parts=["summary"],
            api_endpoint="openai",
            api_key="sk-123",
            model="gpt-4-vision-preview",
            temperature=0.5,
            current_image_input={'base64_data': b64_img, 'mime_type': 'image/png'},
            custom_prompt=None
        )

        kwargs = mock_chat_api_call.call_args.kwargs
        payload = kwargs['messages_payload']
        user_content_parts = payload[0]['content']

        assert len(user_content_parts) == 2  # RAG text + image

        text_part = next(p for p in user_content_parts if p['type'] == 'text')
        image_part = next(p for p in user_content_parts if p['type'] == 'image_url')

        assert "Summary: This is a summary." in text_part['text']
        assert "Describe this." in text_part['text']
        assert image_part['image_url']['url'].startswith("data:image/png;base64,")

    @patch('tldw_chatbook.Chat.Chat_Functions.chat_api_call')
    def test_chat_adapts_payload_for_deepseek(self, mock_chat_api_call):
        chat(
            message="Hello",
            history=[
                {"role": "user", "content": [{"type": "text", "text": "Old message"},
                                             {"type": "image_url", "image_url": {"url": "data:..."}}]},
                {"role": "assistant", "content": "Old reply"}
            ],
            media_content=None,
            selected_parts=[],
            api_endpoint="deepseek",  # The endpoint that needs adaptation
            api_key="sk-123",
            model="deepseek-chat",
            temperature=0.7,
            custom_prompt=None,
            image_history_mode="tag_past"
        )

        kwargs = mock_chat_api_call.call_args.kwargs
        adapted_payload = kwargs['messages_payload']

        # Check that all content fields are strings, not lists of parts
        assert isinstance(adapted_payload[0]['content'], str)
        assert adapted_payload[0]['content'] == "Old message\n<image: prior_history.image>"
        assert isinstance(adapted_payload[1]['content'], str)
        assert adapted_payload[1]['content'] == "Old reply"
        assert isinstance(adapted_payload[2]['content'], str)
        assert adapted_payload[2]['content'] == "Hello"


@pytest.mark.integration
class TestChatHistorySaving:
    def test_save_chat_history_to_db_new_conversation(self, db_instance: CharactersRAGDB):
        # Create default character first
        default_char_id = db_instance.add_character_card({
            "name": "Default Character",
            "description": "Default character for general conversations",
            "system_prompt": "You are a helpful assistant.",
            "tags": ["default"],
            "creator_notes": ""
        })
        
        # The history format is now OpenAI's message objects
        chatbot_history = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "General Kenobi"}
        ]

        # Uses default character
        conv_id, status = save_chat_history_to_db_wrapper(
            db=db_instance,
            chatbot_history=chatbot_history,
            conversation_id=None,
            media_content_for_char_assoc=None,
            character_name_for_chat=None
        )

        assert "success" in status.lower()
        assert conv_id is not None

        messages = db_instance.get_messages_for_conversation(conv_id)
        assert len(messages) == 2
        assert messages[0]['sender'] == 'user'
        assert messages[1]['sender'] == 'assistant'

        conv_details = db_instance.get_conversation_by_id(conv_id)
        assert conv_details['character_id'] == default_char_id  # Default character

    def test_save_chat_history_with_image(self, db_instance: CharactersRAGDB):
        # Create default character first
        db_instance.add_character_card({
            "name": "Default Character",
            "description": "Default character for general conversations",
            "system_prompt": "You are a helpful assistant.",
            "tags": ["default"],
            "creator_notes": ""
        })
        
        b64_img = create_base64_image()
        chatbot_history = [
            {"role": "user", "content": [
                {"type": "text", "text": "Look at this image"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
            ]},
            {"role": "assistant", "content": "I see a 1x1 black square."}
        ]

        conv_id, status = save_chat_history_to_db_wrapper(db_instance, chatbot_history, None, None, None)
        assert "success" in status.lower()

        messages = db_instance.get_messages_for_conversation(conv_id)
        assert len(messages) == 2
        assert messages[0]['content'] == "Look at this image"
        assert messages[0]['image_data'] is not None
        assert messages[0]['image_mime_type'] == "image/png"
        assert messages[1]['image_data'] is None

    def test_resave_chat_history(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "Resaver"})
        initial_history = [{"role": "user", "content": "First message"}]
        conv_id, _ = save_chat_history_to_db_wrapper(db_instance, initial_history, None, None, "Resaver")

        updated_history = [
            {"role": "user", "content": "New first message"},
            {"role": "assistant", "content": "New reply"}
        ]

        # Resave with same conv_id
        resave_id, status = save_chat_history_to_db_wrapper(db_instance, updated_history, conv_id, None, "Resaver")
        assert "success" in status.lower()
        assert resave_id == conv_id

        messages = db_instance.get_messages_for_conversation(conv_id)
        assert len(messages) == 2
        assert messages[0]['content'] == "New first message"


@pytest.mark.integration
class TestCharacterManagement:
    def test_save_and_load_character(self, db_instance: CharactersRAGDB):
        char_data = {
            "name": "Super Coder",
            "description": "A character that codes.",
            "image": create_base64_image()
        }

        char_id = save_character(db_instance, char_data)
        assert isinstance(char_id, int)

        loaded_chars = load_characters(db_instance)
        assert "Super Coder" in loaded_chars
        loaded_char_data = loaded_chars["Super Coder"]
        assert loaded_char_data['description'] == "A character that codes."
        assert loaded_char_data['image_base64'] is not None

    def test_get_character_names(self, db_instance: CharactersRAGDB):
        save_character(db_instance, {"name": "Beta"})
        save_character(db_instance, {"name": "Alpha"})

        # Default Assistant is created during DB initialization
        names = get_character_names(db_instance)
        assert names == ["Alpha", "Beta", "Default Assistant"]


@pytest.mark.unit
class TestChatDictionary:
    @patch('tldw_chatbook.Chat.Chat_Functions.validate_path')
    def test_parse_user_dict_markdown_file(self, mock_validate_path, tmp_path):
        # Mock validate_path to return the validated path
        dict_file = tmp_path / "test_dict.md"
        mock_validate_path.return_value = str(dict_file)
        
        dict_content = """key1: value1
key2: |
  This is a
  multiline value.
---@@@---
/key3/i: value3
"""
        dict_file.write_text(dict_content)

        parsed = parse_user_dict_markdown_file(str(dict_file))
        assert parsed["key1"] == "value1"
        assert parsed["key2"] == "This is a\n  multiline value."
        assert parsed["/key3/i"] == "value3"

    def test_process_user_input_simple_replacement(self):
        entries = [ChatDictionary(key="hello", content="GREETING")]
        user_input = "I said hello to the world."
        result = process_user_input(user_input, entries)
        assert result == "I said GREETING to the world."

    def test_process_user_input_regex_replacement(self):
        entries = [ChatDictionary(key=r"/h[aeiou]llo/i", content="GREETING")]
        user_input = "I said hallo and heLlo."
        # It replaces only the first match
        result = process_user_input(user_input, entries)
        assert result == "I said GREETING and heLlo."

    def test_process_user_input_token_budget(self):
        # Content is 4 tokens, budget is 3. Should not replace.
        entries = [ChatDictionary(key="long", content="this is too long")]
        user_input = "This is a long test."
        result = process_user_input(user_input, entries, max_tokens=3)
        assert result == "This is a long test."

        # Content is 3 tokens, budget is 3. Should replace.
        entries = [ChatDictionary(key="short", content="this is fine")]
        user_input = "This is a short test."
        result = process_user_input(user_input, entries, max_tokens=3)
        assert result == "This is a this is fine test."

#
# End of test_chat_functions.py
########################################################################################################################