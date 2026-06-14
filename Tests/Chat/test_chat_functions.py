# test_chat_functions.py
#
# Imports
import pytest
import base64
import io
import json
from pathlib import Path
from unittest.mock import patch, MagicMock
#
# 3rd-party Libraries
import requests
from PIL import Image
#
# Local Imports
from tldw_chatbook.DB.ChaChaNotes_DB import CharactersRAGDB, ConflictError, InputError
import tldw_chatbook.Chat.Chat_Functions as chat_functions_module
import tldw_chatbook.LLM_Calls.LLM_API_Calls as llm_api_calls_module
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

DUMMY_OPENAI_API_KEY = "DUMMY_OPENAI_API_KEY"
DUMMY_ANTHROPIC_API_KEY = "DUMMY_ANTHROPIC_API_KEY"


def test_huggingface_chat_api_call_passes_max_tokens_to_adapter(monkeypatch):
    captured_kwargs = {}

    def fake_huggingface_handler(**kwargs):
        captured_kwargs.update(kwargs)
        return "OK"

    monkeypatch.setitem(
        chat_functions_module.API_CALL_HANDLERS,
        "huggingface",
        fake_huggingface_handler,
    )

    chat_functions_module.chat_api_call(
        api_endpoint="huggingface",
        api_key="hf-key",
        messages_payload=[{"role": "user", "content": "hello"}],
        model="org/model",
        max_tokens=12,
    )

    assert captured_kwargs["max_tokens"] == 12
    assert "max_new_tokens" not in captured_kwargs


def test_chat_api_call_does_not_log_api_key_fragments(monkeypatch):
    captured_logs = []
    sink_id = chat_functions_module.logger.add(
        lambda message: captured_logs.append(str(message)),
        level="DEBUG",
    )

    def fake_openai_handler(**_kwargs):
        return "OK"

    monkeypatch.setitem(
        chat_functions_module.API_CALL_HANDLERS,
        "openai",
        fake_openai_handler,
    )

    secret = "prefix-secret-middle-secret-suffix"
    try:
        chat_functions_module.chat_api_call(
            api_endpoint="openai",
            api_key=secret,
            messages_payload=[{"role": "user", "content": "hello"}],
            model="gpt-test",
        )
    finally:
        chat_functions_module.logger.remove(sink_id)

    rendered_logs = "".join(captured_logs)
    assert secret not in rendered_logs
    assert secret[:6] not in rendered_logs
    assert secret[-6:] not in rendered_logs


def test_chat_provider_adapters_do_not_log_api_key_fragments():
    modules = [
        chat_functions_module,
        llm_api_calls_module,
    ]
    suspicious_lines = []
    for module in modules:
        source_path = Path(module.__file__)
        for line_number, line in enumerate(source_path.read_text().splitlines(), start=1):
            if "API Key" not in line:
                continue
            if "..." in line or "[:" in line or "log_key" in line:
                suspicious_lines.append(f"{source_path.name}:{line_number}: {line.strip()}")

    assert suspicious_lines == []


def create_base64_image():
    """Creates a dummy 1x1 png and returns its base64 string."""
    img_bytes = io.BytesIO()
    Image.new('RGB', (1, 1)).save(img_bytes, format='PNG')
    return base64.b64encode(img_bytes.getvalue()).decode('utf-8')


class _CapturedSession:
    """Small requests.Session stand-in that records the outbound JSON payload."""

    def __init__(self, captured, response_data):
        self._captured = captured
        self._response_data = response_data
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *_exc_info):
        self.close()
        return False

    def mount(self, *_args, **_kwargs):
        return None

    def close(self):
        self.closed = True
        return None

    def post(self, url, *, headers=None, json=None, stream=False, timeout=None):
        self._captured.update(
            {
                "url": url,
                "headers": headers,
                "json": json,
                "stream": stream,
                "timeout": timeout,
            }
        )
        return _FakeProviderResponse(self._response_data, session=self)


class _FakeProviderResponse:
    status_code = 200
    text = "{}"

    def __init__(self, response_data, session=None):
        self._response_data = response_data
        self._session = session

    def raise_for_status(self):
        return None

    def json(self):
        return self._response_data

    def iter_lines(self, decode_unicode=False):
        if self._session is not None and self._session.closed:
            raise requests.exceptions.ConnectionError("session closed before stream consumption")
        lines = self._response_data if isinstance(self._response_data, list) else []
        for line in lines:
            yield line if decode_unicode else line.encode("utf-8")

    def close(self):
        return None


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

    def test_openai_generation_reasoning_params_are_mapped(self, mock_handlers, mocker):
        mock_openai_handler = mocker.MagicMock(return_value="OpenAI response")
        mock_openai_handler.__name__ = "mock_openai_handler"
        mock_handlers.get.return_value = mock_openai_handler

        chat_api_call(
            api_endpoint="openai",
            messages_payload=[{"role": "user", "content": "test"}],
            model="o3",
            seed=123,
            presence_penalty=0.2,
            frequency_penalty=0.3,
            reasoning_effort="high",
            reasoning_summary="auto",
            verbosity="medium",
        )

        kwargs = mock_openai_handler.call_args.kwargs
        assert kwargs["model"] == "o3"
        assert kwargs["seed"] == 123
        assert kwargs["presence_penalty"] == 0.2
        assert kwargs["frequency_penalty"] == 0.3
        assert kwargs["reasoning_effort"] == "high"
        assert kwargs["reasoning_summary"] == "auto"
        assert kwargs["verbosity"] == "medium"

    def test_anthropic_thinking_params_are_mapped(self, mock_handlers, mocker):
        mock_anthropic_handler = mocker.MagicMock(return_value="Anthropic response")
        mock_anthropic_handler.__name__ = "mock_anthropic_handler"
        mock_handlers.get.return_value = mock_anthropic_handler

        chat_api_call(
            api_endpoint="anthropic",
            messages_payload=[{"role": "user", "content": "test"}],
            model="claude-sonnet-4-20250514",
            thinking_effort="high",
            thinking_budget_tokens=4096,
        )

        kwargs = mock_anthropic_handler.call_args.kwargs
        assert kwargs["model"] == "claude-sonnet-4-20250514"
        assert kwargs["thinking_effort"] == "high"
        assert kwargs["thinking_budget_tokens"] == 4096

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


@pytest.mark.unit
class TestProviderRequestPayloads:
    def test_openai_reasoning_uses_responses_api_and_normalizes_output(self, monkeypatch):
        from tldw_chatbook.LLM_Calls import LLM_API_Calls

        captured = {}
        response_data = {
            "id": "resp_test",
            "created_at": 123,
            "model": "o3",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "reasoned answer"}],
                }
            ],
            "usage": {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11},
        }
        monkeypatch.setattr(
            LLM_API_Calls,
            "load_settings",
            lambda: {"openai_api": {"api_base_url": "https://api.openai.test/v1"}},
        )
        monkeypatch.setattr(
            LLM_API_Calls.requests,
            "Session",
            lambda: _CapturedSession(captured, response_data),
        )

        response = LLM_API_Calls.chat_with_openai(
            input_data=[{"role": "user", "content": "test"}],
            api_key=DUMMY_OPENAI_API_KEY,
            model="o3",
            streaming=False,
            temp=0.3,
            maxp=0.8,
            max_tokens=512,
            reasoning_effort="high",
            reasoning_summary="auto",
            verbosity="medium",
        )

        assert captured["url"] == "https://api.openai.test/v1/responses"
        assert captured["json"]["input"] == [{"role": "user", "content": "test"}]
        assert "messages" not in captured["json"]
        assert captured["json"]["reasoning"] == {"effort": "high", "summary": "auto"}
        assert captured["json"]["text"] == {"verbosity": "medium"}
        assert captured["json"]["max_output_tokens"] == 512
        assert response["choices"][0]["message"]["content"] == "reasoned answer"
        assert response["usage"] == {"input_tokens": 5, "output_tokens": 6, "total_tokens": 11}

    def test_openai_reasoning_stream_normalizes_responses_events(self, monkeypatch):
        from tldw_chatbook.LLM_Calls import LLM_API_Calls

        captured = {}
        response_events = [
            "data: "
            + json.dumps({"type": "response.output_text.delta", "delta": "streamed answer"}),
            "data: " + json.dumps({"type": "response.completed"}),
        ]
        monkeypatch.setattr(
            LLM_API_Calls,
            "load_settings",
            lambda: {"openai_api": {"api_base_url": "https://api.openai.test/v1"}},
        )
        monkeypatch.setattr(
            LLM_API_Calls.requests,
            "Session",
            lambda: _CapturedSession(captured, response_events),
        )

        stream = LLM_API_Calls.chat_with_openai(
            input_data=[{"role": "user", "content": "test"}],
            api_key=DUMMY_OPENAI_API_KEY,
            model="o3",
            streaming=True,
            reasoning_effort="low",
        )

        chunks = list(stream)
        combined = "".join(chunks)

        assert captured["url"] == "https://api.openai.test/v1/responses"
        assert captured["stream"] is True
        assert "chat.completion.chunk" in combined
        assert "streamed answer" in combined
        assert "response.output_text.delta" not in combined
        assert chunks[-1] == "data: [DONE]\n\n"

    def test_huggingface_v1_base_url_uses_chat_completions_path_once(self, monkeypatch):
        from tldw_chatbook.LLM_Calls import LLM_API_Calls

        captured = {}
        response_data = {
            "id": "hf_test",
            "choices": [{"message": {"content": "OK"}}],
        }
        monkeypatch.setattr(
            LLM_API_Calls,
            "load_settings",
            lambda: {
                "huggingface_api": {
                    "api_base_url": "https://router.huggingface.co/v1",
                    "api_timeout": 30,
                    "api_retries": 0,
                }
            },
        )
        monkeypatch.setattr(
            LLM_API_Calls.requests,
            "Session",
            lambda: _CapturedSession(captured, response_data),
        )

        response = LLM_API_Calls.chat_with_huggingface(
            input_data=[{"role": "user", "content": "test"}],
            api_key="hf-key",
            model="openai/gpt-oss-120b",
            streaming=False,
            max_tokens=8,
        )

        assert captured["url"] == "https://router.huggingface.co/v1/chat/completions"
        assert captured["json"]["model"] == "openai/gpt-oss-120b"
        assert captured["json"]["max_tokens"] == 8
        assert response == response_data

    def test_huggingface_debug_logs_do_not_include_api_key(self, monkeypatch):
        from tldw_chatbook.LLM_Calls import LLM_API_Calls

        captured = {}
        debug_messages = []
        response_data = {
            "id": "hf_test",
            "choices": [{"message": {"content": "OK"}}],
        }
        monkeypatch.setattr(
            LLM_API_Calls,
            "load_settings",
            lambda: {
                "huggingface_api": {
                    "api_base_url": "https://router.huggingface.co/v1",
                    "api_chat_path": "chat/completions",
                    "api_timeout": 30,
                    "api_retries": 0,
                }
            },
        )
        monkeypatch.setattr(
            LLM_API_Calls.requests,
            "Session",
            lambda: _CapturedSession(captured, response_data),
        )
        monkeypatch.setattr(
            LLM_API_Calls.logger,
            "debug",
            lambda message, *args, **kwargs: debug_messages.append(str(message)),
        )

        LLM_API_Calls.chat_with_huggingface(
            input_data=[{"role": "user", "content": "test"}],
            api_key="hf-secret-value",
            model="openai/gpt-oss-120b",
            streaming=False,
            max_tokens=8,
        )

        combined_debug = "\n".join(debug_messages)
        assert "hf-secret-value" not in combined_debug
        assert "Bearer hf-secret-value" not in combined_debug

    def test_anthropic_thinking_omits_incompatible_sampling_params(self, monkeypatch):
        from tldw_chatbook.LLM_Calls import LLM_API_Calls

        captured = {}
        monkeypatch.setattr(
            LLM_API_Calls,
            "load_settings",
            lambda: {"anthropic_api": {"api_base_url": "https://api.anthropic.test/v1"}},
        )
        monkeypatch.setattr(
            LLM_API_Calls.requests,
            "Session",
            lambda: _CapturedSession(
                captured,
                {
                    "id": "msg_test",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "thinking answer"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 5},
                },
            ),
        )

        LLM_API_Calls.chat_with_anthropic(
            input_data=[{"role": "user", "content": "test"}],
            api_key=DUMMY_ANTHROPIC_API_KEY,
            model="claude-sonnet-4-20250514",
            streaming=False,
            temp=0.2,
            topp=0.8,
            topk=40,
            max_tokens=4096,
            thinking_budget_tokens=1024,
        )

        assert captured["json"]["thinking"] == {"type": "enabled", "budget_tokens": 1024}
        assert "temperature" not in captured["json"]
        assert "top_p" not in captured["json"]
        assert "top_k" not in captured["json"]

    def test_anthropic_thinking_effort_maps_to_budget_tokens(self, monkeypatch):
        from tldw_chatbook.LLM_Calls import LLM_API_Calls

        captured = {}
        monkeypatch.setattr(
            LLM_API_Calls,
            "load_settings",
            lambda: {"anthropic_api": {"api_base_url": "https://api.anthropic.test/v1"}},
        )
        monkeypatch.setattr(
            LLM_API_Calls.requests,
            "Session",
            lambda: _CapturedSession(
                captured,
                {
                    "id": "msg_test",
                    "model": "claude-sonnet-4-20250514",
                    "content": [{"type": "text", "text": "thinking answer"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 5},
                },
            ),
        )

        response = LLM_API_Calls.chat_with_anthropic(
            input_data=[{"role": "user", "content": "test"}],
            api_key=DUMMY_ANTHROPIC_API_KEY,
            model="claude-sonnet-4-20250514",
            streaming=False,
            max_tokens=12000,
            thinking_effort="high",
        )

        assert captured["url"] == "https://api.anthropic.test/v1/messages"
        assert captured["json"]["thinking"] == {"type": "enabled", "budget_tokens": 8192}
        assert captured["json"]["max_tokens"] == 12000
        assert response["choices"][0]["message"]["content"] == "thinking answer"

    def test_anthropic_latest_opus_uses_adaptive_thinking_effort(self, monkeypatch):
        from tldw_chatbook.LLM_Calls import LLM_API_Calls

        captured = {}
        monkeypatch.setattr(
            LLM_API_Calls,
            "load_settings",
            lambda: {"anthropic_api": {"api_base_url": "https://api.anthropic.test/v1"}},
        )
        monkeypatch.setattr(
            LLM_API_Calls.requests,
            "Session",
            lambda: _CapturedSession(
                captured,
                {
                    "id": "msg_test",
                    "model": "claude-opus-4-7",
                    "content": [{"type": "text", "text": "adaptive answer"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 5},
                },
            ),
        )

        LLM_API_Calls.chat_with_anthropic(
            input_data=[{"role": "user", "content": "test"}],
            api_key=DUMMY_ANTHROPIC_API_KEY,
            model="claude-opus-4-7",
            streaming=False,
            max_tokens=12000,
            thinking_effort="xhigh",
            thinking_budget_tokens=4096,
        )

        assert captured["json"]["thinking"] == {"type": "adaptive", "effort": "xhigh"}

    def test_anthropic_current_opus_uses_adaptive_thinking_effort(self, monkeypatch):
        from tldw_chatbook.LLM_Calls import LLM_API_Calls

        captured = {}
        monkeypatch.setattr(
            LLM_API_Calls,
            "load_settings",
            lambda: {"anthropic_api": {"api_base_url": "https://api.anthropic.test/v1"}},
        )
        monkeypatch.setattr(
            LLM_API_Calls.requests,
            "Session",
            lambda: _CapturedSession(
                captured,
                {
                    "id": "msg_test",
                    "model": "claude-opus-4-8",
                    "content": [{"type": "text", "text": "adaptive answer"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 4, "output_tokens": 5},
                },
            ),
        )

        LLM_API_Calls.chat_with_anthropic(
            input_data=[{"role": "user", "content": "test"}],
            api_key=DUMMY_ANTHROPIC_API_KEY,
            model="claude-opus-4-8",
            streaming=False,
            max_tokens=12000,
            thinking_effort="high",
        )

        assert captured["json"]["thinking"] == {"type": "adaptive", "effort": "high"}


@pytest.mark.integration
class TestChatHistorySaving:
    def test_save_chat_history_to_db_new_conversation(self, db_instance: CharactersRAGDB):
        # The history format is now OpenAI's message objects
        chatbot_history = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "General Kenobi"}
        ]

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
        assert conv_details['character_id'] is None
        assert conv_details['assistant_kind'] is None
        assert conv_details['title'] == "New Chat"

    def test_save_chat_history_with_image(self, db_instance: CharactersRAGDB):
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

    def test_save_character_chat_history_uses_canonical_character_id_as_assistant_id(self, db_instance: CharactersRAGDB):
        character_id = db_instance.add_character_card({"name": "Canonical Saver"})

        conv_id, status = save_chat_history_to_db_wrapper(
            db=db_instance,
            chatbot_history=[
                {"role": "user", "content": "Hello there"},
                {"role": "assistant", "content": "General Kenobi"},
            ],
            conversation_id=None,
            media_content_for_char_assoc=None,
            character_name_for_chat="Canonical Saver",
        )

        assert "success" in status.lower()
        assert conv_id is not None

        conv_details = db_instance.get_conversation_by_id(conv_id)
        assert conv_details["assistant_kind"] == "character"
        assert conv_details["assistant_id"] == str(character_id)
        assert conv_details["title"] == "Chat with Canonical Saver"

    def test_resave_chat_history(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "Resaver"})
        initial_history = [
            {
                "id": "msg-user-1",
                "role": "user",
                "content": "First message",
            },
            {
                "id": "msg-assistant-1",
                "role": "assistant",
                "content": "Initial reply",
                "parent_message_id": "msg-user-1",
                "feedback": "liked",
            },
        ]
        conv_id, _ = save_chat_history_to_db_wrapper(db_instance, initial_history, None, None, "Resaver")

        db_instance.create_message_variant(
            original_message_id="msg-assistant-1",
            variant_content="Variant reply",
            is_selected=True,
        )

        before_messages = {
            message["id"]: message
            for message in db_instance.get_messages_for_conversation(conv_id)
        }

        updated_history = [
            {
                "id": "msg-user-1",
                "role": "user",
                "content": "New first message",
            },
            {
                "id": "msg-assistant-1",
                "role": "assistant",
                "content": "New reply",
                "parent_message_id": "msg-user-1",
                "feedback": "liked",
            },
        ]

        # Resave with same conv_id
        resave_id, status = save_chat_history_to_db_wrapper(db_instance, updated_history, conv_id, None, "Resaver")
        assert "success" in status.lower()
        assert resave_id == conv_id

        messages = {
            message["id"]: message
            for message in db_instance.get_messages_for_conversation(conv_id)
        }
        assert set(messages) == set(before_messages)
        assert messages["msg-user-1"]["content"] == "New first message"
        assert messages["msg-assistant-1"]["content"] == "New reply"
        assert messages["msg-assistant-1"]["parent_message_id"] == before_messages["msg-assistant-1"]["parent_message_id"]
        assert messages["msg-assistant-1"]["feedback"] == before_messages["msg-assistant-1"]["feedback"]
        assert messages["msg-assistant-1"]["variant_of"] == before_messages["msg-assistant-1"]["variant_of"]
        assert messages["msg-assistant-1"]["variant_number"] == before_messages["msg-assistant-1"]["variant_number"]
        assert messages["msg-assistant-1"]["is_selected_variant"] == before_messages["msg-assistant-1"]["is_selected_variant"]
        assert messages["msg-assistant-1"]["total_variants"] == before_messages["msg-assistant-1"]["total_variants"]

    def test_resave_chat_history_preserves_backend_while_normalizing_stale_identity_metadata(self, db_instance: CharactersRAGDB):
        char_id = db_instance.add_character_card({"name": "Resaver"})
        conversation_id = db_instance.add_conversation(
            {
                "character_id": char_id,
                "assistant_kind": "character",
                "assistant_id": "Resaver",
                "runtime_backend": "server",
                "discovery_owner": "general_chat",
                "discovery_entity_id": "legacy.display.name",
                "title": "Chat with Resaver",
                "client_id": db_instance.client_id,
            }
        )

        resave_id, status = save_chat_history_to_db_wrapper(
            db_instance,
            [
                {"role": "user", "content": "New first message"},
                {"role": "assistant", "content": "New reply"},
            ],
            conversation_id,
            None,
            "Resaver",
        )

        assert "success" in status.lower()
        assert resave_id == conversation_id

        updated_conversation = db_instance.get_conversation_by_id(conversation_id)
        assert updated_conversation["assistant_kind"] == "character"
        assert updated_conversation["assistant_id"] == str(char_id)
        assert updated_conversation["runtime_backend"] == "server"
        assert updated_conversation["discovery_owner"] == "ccp_character"
        assert updated_conversation["discovery_entity_id"] == str(char_id)
        assert updated_conversation["title"] == "Chat with Resaver"

    def test_resave_chat_history_rejects_generic_context_for_character_conversation(self, db_instance: CharactersRAGDB):
        db_instance.add_character_card({"name": "Resaver"})
        conversation_id, status = save_chat_history_to_db_wrapper(
            db_instance,
            [{"role": "user", "content": "Bound message"}],
            None,
            None,
            "Resaver",
        )
        assert "success" in status.lower()

        resave_id, resave_status = save_chat_history_to_db_wrapper(
            db_instance,
            [{"role": "user", "content": "Generic overwrite attempt"}],
            conversation_id,
            None,
            None,
        )

        assert resave_id == conversation_id
        assert "mismatch" in resave_status.lower()
        messages = db_instance.get_messages_for_conversation(conversation_id)
        assert [message["content"] for message in messages] == ["Bound message"]

    def test_resave_chat_history_rejects_character_context_for_generic_conversation(self, db_instance: CharactersRAGDB):
        db_instance.add_character_card({"name": "Resaver"})
        conversation_id, status = save_chat_history_to_db_wrapper(
            db_instance,
            [{"role": "user", "content": "Generic message"}],
            None,
            None,
            None,
        )
        assert "success" in status.lower()

        resave_id, resave_status = save_chat_history_to_db_wrapper(
            db_instance,
            [{"role": "user", "content": "Character overwrite attempt"}],
            conversation_id,
            None,
            "Resaver",
        )

        assert resave_id == conversation_id
        assert "mismatch" in resave_status.lower()
        messages = db_instance.get_messages_for_conversation(conversation_id)
        assert [message["content"] for message in messages] == ["Generic message"]

    def test_save_chat_history_creates_persona_backed_conversation_when_metadata_is_explicit(self, db_instance: CharactersRAGDB):
        conversation_id, status = save_chat_history_to_db_wrapper(
            db=db_instance,
            chatbot_history=[
                {"role": "user", "content": "Hello persona"},
                {"role": "assistant", "content": "Hello human"},
            ],
            conversation_id=None,
            media_content_for_char_assoc=None,
            character_name_for_chat=None,
            assistant_kind="persona",
            assistant_id="persona.local.alice",
            persona_memory_mode="read_write",
            runtime_backend="server",
            discovery_owner="ccp_persona",
            discovery_entity_id="persona.local.alice",
        )

        assert "success" in status.lower()
        assert conversation_id is not None

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation["character_id"] is None
        assert conversation["assistant_kind"] == "persona"
        assert conversation["assistant_id"] == "persona.local.alice"
        assert conversation["persona_memory_mode"] == "read_write"
        assert conversation["runtime_backend"] == "server"
        assert conversation["discovery_owner"] == "ccp_persona"
        assert conversation["discovery_entity_id"] == "persona.local.alice"
        assert conversation["title"] == "Chat with persona.local.alice"

    def test_resave_chat_history_preserves_existing_persona_metadata(self, db_instance: CharactersRAGDB):
        conversation_id = db_instance.add_conversation(
            {
                "assistant_kind": "persona",
                "assistant_id": "persona.local.alice",
                "persona_memory_mode": "read_only",
                "runtime_backend": "server",
                "discovery_owner": "ccp_persona",
                "discovery_entity_id": "persona.local.alice",
                "title": "Chat with persona.local.alice",
                "client_id": db_instance.client_id,
            }
        )

        resave_id, status = save_chat_history_to_db_wrapper(
            db=db_instance,
            chatbot_history=[
                {"role": "user", "content": "Persona conversation"},
                {"role": "assistant", "content": "Persona reply"},
            ],
            conversation_id=conversation_id,
            media_content_for_char_assoc=None,
            character_name_for_chat=None,
        )

        assert "success" in status.lower()
        assert resave_id == conversation_id

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation["assistant_kind"] == "persona"
        assert conversation["assistant_id"] == "persona.local.alice"
        assert conversation["persona_memory_mode"] == "read_only"
        assert conversation["runtime_backend"] == "server"
        assert conversation["discovery_owner"] == "ccp_persona"
        assert conversation["discovery_entity_id"] == "persona.local.alice"

    def test_resave_generic_chat_history_preserves_existing_runtime_and_scope_metadata(self, db_instance: CharactersRAGDB):
        conversation_id = db_instance.add_conversation(
            {
                "title": "Workspace Generic",
                "runtime_backend": "server",
                "discovery_owner": "general_chat",
                "scope_type": "workspace",
                "workspace_id": "ws-1",
                "client_id": db_instance.client_id,
            }
        )

        resave_id, status = save_chat_history_to_db_wrapper(
            db=db_instance,
            chatbot_history=[
                {"role": "user", "content": "Workspace conversation"},
                {"role": "assistant", "content": "Workspace reply"},
            ],
            conversation_id=conversation_id,
            media_content_for_char_assoc=None,
            character_name_for_chat=None,
        )

        assert "success" in status.lower()
        assert resave_id == conversation_id

        conversation = db_instance.get_conversation_by_id(conversation_id)
        assert conversation["assistant_kind"] is None
        assert conversation["assistant_id"] is None
        assert conversation["runtime_backend"] == "server"
        assert conversation["discovery_owner"] == "general_chat"
        assert conversation["scope_type"] == "workspace"
        assert conversation["workspace_id"] == "ws-1"


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
    @patch('tldw_chatbook.Character_Chat.Chat_Dictionary_Lib.validate_path')
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
