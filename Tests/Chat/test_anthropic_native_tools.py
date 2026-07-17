"""
Tests for Anthropic native tool-calls request-side conversion (task-263 AC#2),
non-streaming response-side normalization (task-263 AC#1a), and streaming
response-side normalization (task-263 AC#1b).

`chat_with_anthropic` must convert OpenAI-shaped ``tools`` and OpenAI-shaped
tool-call history (assistant ``tool_calls`` + ``role="tool"`` messages) into
Anthropic's native ``tool_use`` / ``tool_result`` block format before sending
the request, and must normalize Anthropic ``tool_use`` content blocks in the
response back into OpenAI-shaped ``message.tool_calls`` -- both for the full
non-streaming response and, fragment by fragment, for the streaming SSE
generator (``content_block_start``/``input_json_delta`` -> OpenAI
``delta.tool_calls`` fragments).

Mirrors the mocking pattern from
`Tests/Chat/test_chat_mocked_apis.py::test_anthropic_chat_mocked`: patch
`requests.Session.post`, drive the real dispatcher via `chat_api_call`, and
inspect the JSON payload actually sent (or the normalized response returned).
"""

import json
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


def _anthropic_tool_use_response():
    return {"id": "msg_2", "type": "message", "role": "assistant", "model": "claude-x",
            "content": [
                {"type": "text", "text": "Checking."},
                {"type": "tool_use", "id": "toolu_X", "name": "calculator",
                 "input": {"expression": "2+2"}}],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 9}}


def _call_anthropic_get_result(mock_post, response_json, messages, **extra):
    """Like `_call_anthropic` but returns the normalized `chat_api_call`
    result instead of the request JSON that was sent."""
    mock_response = Mock()
    mock_response.json.return_value = response_json
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    return chat_api_call(
        "anthropic",
        messages_payload=messages,
        api_key="test-key",
        model="claude-3-opus-20240229",
        streaming=False,
        **extra,
    )


@patch("requests.Session.post")
def test_tool_use_response_normalizes_to_openai_tool_calls(mock_post):
    result = _call_anthropic_get_result(
        mock_post, _anthropic_tool_use_response(),
        [{"role": "user", "content": "2+2?"}],
    )
    message = result["choices"][0]["message"]

    assert result["choices"][0]["finish_reason"] == "tool_calls"
    assert message["content"] == "Checking."
    assert message["tool_calls"] == [{
        "id": "toolu_X", "type": "function",
        "function": {"name": "calculator",
                     "arguments": json.dumps({"expression": "2+2"})}}]


@patch("requests.Session.post")
def test_text_only_response_has_no_tool_calls_key(mock_post):
    result = _call_anthropic_get_result(
        mock_post, _anthropic_text_response(),
        [{"role": "user", "content": "hi"}],
    )
    message = result["choices"][0]["message"]

    assert "tool_calls" not in message


@patch("requests.Session.post")
def test_tool_use_only_response_has_empty_content_and_tool_calls(mock_post):
    """T2 review Minor: a response carrying ONLY tool_use blocks (no text
    parts) must normalize to content == "" with tool_calls present and
    finish_reason "tool_calls"."""
    response = {"id": "msg_4", "type": "message", "role": "assistant",
                "model": "claude-x",
                "content": [{"type": "tool_use", "id": "toolu_O",
                             "name": "calculator",
                             "input": {"expression": "1+1"}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 2, "output_tokens": 3}}
    result = _call_anthropic_get_result(
        mock_post, response, [{"role": "user", "content": "1+1?"}])
    choice = result["choices"][0]
    assert choice["message"]["content"] == ""
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"][0]["id"] == "toolu_O"


@patch("requests.Session.post")
def test_junk_tool_use_blocks_normalize_parseably(mock_post):
    """T2 review Minor: response-side junk — a tool_use block with input
    None must emit arguments "{}" (never "null"); a nameless block emits an
    empty-name entry that downstream parse_native_tool_calls DROPS without
    crashing (pinned here end-to-end)."""
    from tldw_chatbook.Agents.native_tools import parse_native_tool_calls
    response = {"id": "msg_5", "type": "message", "role": "assistant",
                "model": "claude-x",
                "content": [
                    {"type": "tool_use", "id": "toolu_N",
                     "name": "calculator", "input": None},
                    {"type": "tool_use", "id": "toolu_E", "name": "",
                     "input": {"x": 1}}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 2, "output_tokens": 3}}
    result = _call_anthropic_get_result(
        mock_post, response, [{"role": "user", "content": "go"}])
    entries = result["choices"][0]["message"]["tool_calls"]
    assert entries[0]["function"]["arguments"] == "{}"
    parsed = parse_native_tool_calls(result["choices"][0]["message"])
    assert [(c.name, c.args, c.call_id) for c in parsed] == [
        ("calculator", {}, "toolu_N")]  # nameless entry dropped, no crash


def _anthropic_sse_lines():
    """A realistic Anthropic streaming transcript: a text block (index 0)
    followed by a tool_use block (index 1) whose input arrives across two
    `input_json_delta` fragments, then the terminal `tool_use` stop reason."""
    events = [
        {"type": "message_start", "message": {"id": "msg_3"}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "Checking."}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1,
         "content_block": {"type": "tool_use", "id": "toolu_S",
                           "name": "calculator", "input": {}}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": '{"expres'}},
        {"type": "content_block_delta", "index": 1,
         "delta": {"type": "input_json_delta", "partial_json": 'sion": "2+2"}'}},
        {"type": "content_block_stop", "index": 1},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
        {"type": "message_stop"},
    ]
    return [f"data: {json.dumps(e)}".encode() for e in events]


def _call_anthropic_stream(mock_post, sse_lines, messages, **extra):
    """Streaming counterpart of `_call_anthropic_get_result`: stub a
    streaming Anthropic response (`iter_lines` yields raw SSE byte-lines)
    and return the consumed generator's yielded SSE strings."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = sse_lines
    mock_response.close = Mock()
    mock_post.return_value = mock_response

    result = chat_api_call(
        "anthropic",
        messages_payload=messages,
        api_key="test-key",
        model="claude-3-opus-20240229",
        streaming=True,
        **extra,
    )
    return list(result)


@patch("requests.Session.post")
def test_streaming_tool_use_emits_openai_delta_fragments(mock_post):
    sse_lines = _call_anthropic_stream(
        mock_post, _anthropic_sse_lines(),
        [{"role": "user", "content": "2+2?"}],
    )

    chunks = []
    for line in sse_lines:
        assert line.startswith("data: ")
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))

    fragments = [c["choices"][0]["delta"]["tool_calls"]
                 for c in chunks if "tool_calls" in c["choices"][0].get("delta", {})]
    assert fragments[0] == [{"index": 0, "id": "toolu_S", "type": "function",
                             "function": {"name": "calculator", "arguments": ""}}]
    assert fragments[1] == [{"index": 0,
                             "function": {"arguments": '{"expres'}}]
    assert fragments[2] == [{"index": 0,
                             "function": {"arguments": 'sion": "2+2"}'}}]
    # text still streams, finish_reason still maps: not every chunk carries a
    # "delta" (a finish_reason-only chunk doesn't), so default defensively.
    texts = [c["choices"][0].get("delta", {}).get("content") for c in chunks]
    assert "Checking." in texts
    finishes = [c["choices"][0].get("finish_reason") for c in chunks]
    assert "tool_calls" in finishes


@patch("requests.Session.post")
def test_streaming_fragments_reassemble_via_gateway_accumulator(mock_post):
    """Cross-layer contract pin: feed this handler's yielded SSE strings
    through the REAL gateway accumulator path (`_decode_stream_item` +
    `_ToolCallAccumulator` from `tldw_chatbook.Chat.console_provider_gateway`,
    task-243) and assert they reassemble into one merged tool call. These are
    private module internals -- importing them directly here is intentional:
    it pins the cross-layer fragment-shape contract between this handler and
    the gateway's accumulator (task-263)."""
    from tldw_chatbook.Chat.console_provider_gateway import (
        _decode_stream_item, _ToolCallAccumulator,
    )

    sse_lines = _call_anthropic_stream(
        mock_post, _anthropic_sse_lines(),
        [{"role": "user", "content": "2+2?"}],
    )

    accumulator = _ToolCallAccumulator()
    for line in sse_lines:
        accumulator.feed_payload(_decode_stream_item(line))

    assert accumulator.calls() == ({
        "id": "toolu_S", "type": "function",
        "function": {"name": "calculator",
                     "arguments": '{"expression": "2+2"}'},
    },)


@patch("requests.Session.post")
def test_openai_tool_with_blank_name_is_not_forwarded(mock_post):
    """PR #659 review: an OpenAI tool entry with a blank name must be
    dropped locally — Anthropic 400s on empty tool names."""
    bad_tools = [{"type": "function", "function": {"name": "  ", "parameters": {}}},
                 OPENAI_TOOLS[0]]
    sent = _call_anthropic(mock_post, [{"role": "user", "content": "hi"}],
                           tools=bad_tools)
    assert [t["name"] for t in sent["tools"]] == ["calculator"]
    assert sent["tools"][0]["input_schema"] == OPENAI_TOOLS[0]["function"]["parameters"]


@patch("requests.Session.post")
def test_list_content_with_tool_calls_keeps_text_parts(mock_post):
    """PR #659 review: list-form (multimodal) assistant content alongside
    tool_calls must keep its text parts in the converted blocks."""
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant",
         "content": [{"type": "text", "text": "Let me check."},
                     {"type": "image_url", "image_url": {"url": "data:x"}}],
         "tool_calls": [{"id": "toolu_L", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "toolu_L", "content": "4"},
    ]
    sent = _call_anthropic(mock_post, messages)
    assistant = sent["messages"][1]
    assert assistant["content"][0] == {"type": "text", "text": "Let me check."}
    assert assistant["content"][-1]["type"] == "tool_use"


@patch("requests.Session.post")
def test_tool_use_stop_reason_without_blocks_downgrades_finish_reason(mock_post):
    """PR #659 review: stop_reason tool_use with NO tool_use blocks must not
    emit the self-contradictory finish_reason="tool_calls" without
    message.tool_calls."""
    response = {"id": "msg_6", "type": "message", "role": "assistant",
                "model": "claude-x",
                "content": [{"type": "text", "text": "hm"}],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 1}}
    result = _call_anthropic_get_result(
        mock_post, response, [{"role": "user", "content": "go"}])
    choice = result["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


@patch("requests.Session.post")
def test_streaming_junk_index_event_is_skipped_not_fatal(mock_post):
    """PR #659 review: a malformed tool_use content_block_start (index None)
    must be skipped without aborting the stream — later text still flows."""
    events = [
        {"type": "content_block_start", "index": None,
         "content_block": {"type": "tool_use", "id": "x", "name": "y"}},
        {"type": "content_block_start", "index": 0,
         "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0,
         "delta": {"type": "text_delta", "text": "still alive"}},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        {"type": "message_stop"},
    ]
    lines = [f"data: {json.dumps(e)}".encode() for e in events]
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = iter(lines)
    mock_post.return_value = mock_response
    gen = chat_api_call("anthropic", messages_payload=[{"role": "user", "content": "go"}],
                        api_key="test-key", model="claude-x", streaming=True)
    chunks = []
    for raw in gen:
        payload = raw.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    texts = [c["choices"][0].get("delta", {}).get("content") for c in chunks]
    assert "still alive" in texts
    assert not any("error" in c for c in chunks)
