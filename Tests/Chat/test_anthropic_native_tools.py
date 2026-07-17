"""
Tests for Anthropic native tool-calls request-side conversion (task-263 AC#2).

`chat_with_anthropic` must convert OpenAI-shaped ``tools`` and OpenAI-shaped
tool-call history (assistant ``tool_calls`` + ``role="tool"`` messages) into
Anthropic's native ``tool_use`` / ``tool_result`` block format before sending
the request.

Mirrors the mocking pattern from
`Tests/Chat/test_chat_mocked_apis.py::test_anthropic_chat_mocked`: patch
`requests.Session.post`, drive the real dispatcher via `chat_api_call`, and
inspect the JSON payload actually sent.
"""

from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def _anthropic_text_response(text="ok"):
    return {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-x",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn", "usage": {"input_tokens": 1, "output_tokens": 1},
    }


OPENAI_TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "Evaluate math.",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}]


def _call_anthropic(mock_post, messages, **extra):
    """Shared mock plumbing: stub a non-streaming Anthropic response and
    drive the real dispatch path, returning the JSON body actually posted."""
    mock_response = Mock()
    mock_response.json.return_value = _anthropic_text_response()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    chat_api_call(
        "anthropic",
        messages_payload=messages,
        api_key="test-key",
        model="claude-3-opus-20240229",
        streaming=False,
        **extra,
    )
    return mock_post.call_args[1]["json"]


@patch("requests.Session.post")
def test_openai_tools_convert_to_anthropic_input_schema(mock_post):
    sent = _call_anthropic(
        mock_post,
        [{"role": "user", "content": "2+2?"}],
        tools=OPENAI_TOOLS,
    )
    assert sent["tools"] == [{
        "name": "calculator",
        "description": "Evaluate math.",
        "input_schema": OPENAI_TOOLS[0]["function"]["parameters"],
    }]


@patch("requests.Session.post")
def test_anthropic_shaped_tools_pass_through_untouched(mock_post):
    native = [{"name": "t", "description": "d", "input_schema": {"type": "object"}}]
    sent = _call_anthropic(
        mock_post,
        [{"role": "user", "content": "hi"}],
        tools=native,
    )
    assert sent["tools"] == native


@patch("requests.Session.post")
def test_openai_tool_history_converts_to_anthropic_blocks(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "toolu_A", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{\"expression\": \"2+2\"}"}},
                        {"id": "toolu_B", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{\"expression\": \"3+3\"}"}}],
         },
        {"role": "tool", "tool_call_id": "toolu_A", "content": "4"},
        {"role": "tool", "tool_call_id": "toolu_B", "content": "6"},
    ]
    sent = _call_anthropic(mock_post, messages)["messages"]

    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] == [
        {"type": "tool_use", "id": "toolu_A", "name": "calculator",
         "input": {"expression": "2+2"}},
        {"type": "tool_use", "id": "toolu_B", "name": "calculator",
         "input": {"expression": "3+3"}}]
    # BOTH tool results coalesce into ONE user turn (Anthropic alternation):
    assert sent[2]["role"] == "user"
    assert sent[2]["content"] == [
        {"type": "tool_result", "tool_use_id": "toolu_A", "content": "4"},
        {"type": "tool_result", "tool_use_id": "toolu_B", "content": "6"}]
    assert len(sent) == 3


@patch("requests.Session.post")
def test_assistant_text_plus_tool_calls_keeps_text_block_first(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "toolu_1", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{\"expression\": \"2+2\"}"}}]},
    ]
    sent = _call_anthropic(mock_post, messages)["messages"]

    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] == [
        {"type": "text", "text": "Let me check."},
        {"type": "tool_use", "id": "toolu_1", "name": "calculator",
         "input": {"expression": "2+2"}}]


@patch("requests.Session.post")
def test_malformed_tool_call_arguments_become_empty_input(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "toolu_1", "type": "function",
                         "function": {"name": "calculator", "arguments": "{broken"}}]},
    ]
    sent = _call_anthropic(mock_post, messages)["messages"]

    assert sent[1]["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {}}]


@patch("requests.Session.post")
def test_plain_chat_payload_unchanged(mock_post):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]
    sent = _call_anthropic(mock_post, messages)

    assert "tools" not in sent
    assert sent["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "there"}]},
    ]


@patch("requests.Session.post")
def test_all_junk_tool_calls_fall_back_to_plain_content(mock_post):
    # The live Anthropic API rejects both an empty "content": [] array and a
    # tool_use block with an empty "name". When every tool_calls entry is
    # junk, the assistant turn must be sent as a normal plain-text message
    # instead of a blocks-only message with no valid tool_use (task-263
    # review).
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "hello",
         "tool_calls": ["junk", {"function": "junk"}, {"function": {"name": ""}}]},
    ]
    sent = _call_anthropic(mock_post, messages)["messages"]

    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] == [{"type": "text", "text": "hello"}]


@patch("requests.Session.post")
def test_junk_tool_call_skipped_among_valid_entries(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [
             "junk",
             {"id": "toolu_1", "type": "function",
              "function": {"name": "calculator",
                            "arguments": "{\"expression\": \"2+2\"}"}},
         ]},
    ]
    sent = _call_anthropic(mock_post, messages)["messages"]

    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] == [
        {"type": "tool_use", "id": "toolu_1", "name": "calculator",
         "input": {"expression": "2+2"}}]
