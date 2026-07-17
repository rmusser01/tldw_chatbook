"""
Tests for Cohere v2 /chat migration and native tool-calls (task-267).

Task 1: `chat_with_cohere` migrates from Cohere v1 `/chat` to v2 `/chat`,
text paths only -- byte-compat pinned. v1's flat `parameter_definitions`
shape cannot express nested JSON Schema (MCP tools inexpressible) and
`tool_results` lives outside the history model with no call ids; v2 is
OpenAI-shaped end-to-end (`messages` array incl. `role:"tool"`, JSON-Schema
`tools`, `tool_calls` with ids, incremental streaming deltas). This file
pins the TEXT-ONLY request/response/streaming behavior before any tool
code lands (Tasks 2-4 extend this file).

Mirrors the mocking pattern from `Tests/Chat/test_google_native_tools.py` /
`test_anthropic_native_tools.py`: patch `requests.Session.post`, drive the
real dispatcher via `chat_api_call`, and inspect the JSON payload actually
sent (or the normalized response/stream returned).
"""

import json
import logging
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call

COHERE_V2_URL = "https://api.cohere.com/v2/chat"


def _cohere_text_response(text="ok", finish_reason="COMPLETE"):
    return {
        "id": "resp_1",
        "message": {"role": "assistant",
                    "content": [{"type": "text", "text": text}]},
        "finish_reason": finish_reason,
        "usage": {"billed_units": {"input_tokens": 1, "output_tokens": 1}},
    }


def _call_cohere(mock_post, messages, **extra):
    """Shared mock plumbing: stub a non-streaming Cohere v2 response and
    drive the real dispatch path, returning the JSON body actually posted."""
    mock_response = Mock()
    mock_response.json.return_value = _cohere_text_response()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    chat_api_call(
        "cohere",
        messages_payload=messages,
        api_key="test-key",
        model="command-a-03-2025",
        streaming=False,
        **extra,
    )
    return mock_post.call_args[1]["json"]


def _call_cohere_get_result(mock_post, response_json, messages, **extra):
    """Like `_call_cohere` but returns the normalized `chat_api_call` result
    instead of the request JSON that was sent."""
    mock_response = Mock()
    mock_response.json.return_value = response_json
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    return chat_api_call(
        "cohere",
        messages_payload=messages,
        api_key="test-key",
        model="command-a-03-2025",
        streaming=False,
        **extra,
    )


def _cohere_stream_lines(events):
    return [f"data: {json.dumps(e)}".encode() for e in events]


def _call_cohere_stream(mock_post, sse_lines, messages, **extra):
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = sse_lines
    mock_response.close = Mock()
    mock_post.return_value = mock_response

    result = chat_api_call(
        "cohere",
        messages_payload=messages,
        api_key="test-key",
        model="command-a-03-2025",
        streaming=True,
        **extra,
    )
    return list(result)


def _decode_sse_chunks(sse_lines):
    chunks = []
    for line in sse_lines:
        assert line.startswith("data: ")
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            continue
        chunks.append(json.loads(payload))
    return chunks


# ---------------------------------------------------------------------------
# (a) non-streaming text-only request: v2 endpoint, messages array, no
# v1-only keys, response normalization + finish_reason mapping.
# ---------------------------------------------------------------------------

@patch("requests.Session.post")
def test_request_hits_v2_chat_endpoint(mock_post):
    _call_cohere(mock_post, [{"role": "user", "content": "hi"}])
    assert mock_post.call_args[0][0] == COHERE_V2_URL


@patch("requests.Session.post")
def test_leading_system_message_becomes_system_role_entry(mock_post):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3+3?"},
    ]
    sent = _call_cohere(mock_post, messages)
    assert sent["messages"] == [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "4"},
        {"role": "user", "content": "3+3?"},
    ]


@patch("requests.Session.post")
def test_system_prompt_param_becomes_system_role_entry(mock_post):
    sent = _call_cohere(
        mock_post, [{"role": "user", "content": "hi"}],
        system_message="Be terse.",
    )
    assert sent["messages"][0] == {"role": "system", "content": "Be terse."}


@patch("requests.Session.post")
def test_no_v1_only_keys_in_payload(mock_post):
    messages = [
        {"role": "system", "content": "Be terse."},
        {"role": "user", "content": "hi"},
    ]
    sent = _call_cohere(mock_post, messages)
    assert "preamble" not in sent
    assert "chat_history" not in sent
    assert "message" not in sent


@patch("requests.Session.post")
def test_non_streaming_text_response_normalizes_to_openai_shape(mock_post):
    result = _call_cohere_get_result(
        mock_post, _cohere_text_response("Hi there", "COMPLETE"),
        [{"role": "user", "content": "hi"}],
    )
    choice = result["choices"][0]
    assert choice["message"]["content"] == "Hi there"
    assert choice["message"]["role"] == "assistant"
    assert choice["finish_reason"] == "stop"
    assert "tool_calls" not in choice["message"]


@patch("requests.Session.post")
def test_max_tokens_finish_reason_maps_to_length(mock_post):
    result = _call_cohere_get_result(
        mock_post, _cohere_text_response("cut off", "MAX_TOKENS"),
        [{"role": "user", "content": "hi"}],
    )
    assert result["choices"][0]["finish_reason"] == "length"


# ---------------------------------------------------------------------------
# (b) streaming: content-delta -> delta.content chunks, message-end -> finish
# ---------------------------------------------------------------------------

@patch("requests.Session.post")
def test_streaming_content_delta_emits_openai_delta_content(mock_post):
    events = [
        {"type": "message-start", "delta": {"message": {"role": "assistant"}}},
        {"type": "content-start", "index": 0},
        {"type": "content-delta", "index": 0,
         "delta": {"message": {"content": {"text": "Hel"}}}},
        {"type": "content-delta", "index": 0,
         "delta": {"message": {"content": {"text": "lo"}}}},
        {"type": "content-end", "index": 0},
        {"type": "message-end", "delta": {"finish_reason": "COMPLETE"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "hi"}])
    chunks = _decode_sse_chunks(raw)
    texts = [c["choices"][0].get("delta", {}).get("content") for c in chunks
             if c["choices"][0].get("delta", {}).get("content")]
    assert texts == ["Hel", "lo"]
    finishes = [c["choices"][0].get("finish_reason") for c in chunks
                if c["choices"][0].get("finish_reason")]
    assert finishes == ["stop"]
    assert raw[-1] == "data: [DONE]\n\n"


@patch("requests.Session.post")
def test_streaming_max_tokens_finish_reason_maps_to_length(mock_post):
    events = [
        {"type": "content-delta", "index": 0,
         "delta": {"message": {"content": {"text": "partial"}}}},
        {"type": "message-end", "delta": {"finish_reason": "MAX_TOKENS"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "hi"}])
    chunks = _decode_sse_chunks(raw)
    finishes = [c["choices"][0].get("finish_reason") for c in chunks
                if c["choices"][0].get("finish_reason")]
    assert finishes == ["length"]


# ---------------------------------------------------------------------------
# (c) params map to v2 names; num_generations (v1-only) dropped w/ debug log
# ---------------------------------------------------------------------------

@patch("requests.Session.post")
def test_params_map_to_v2_names(mock_post):
    sent = _call_cohere(
        mock_post, [{"role": "user", "content": "hi"}],
        temp=0.5, topp=0.8, topk=10, max_tokens=100,
        stop=["END"], seed=42,
        frequency_penalty=0.1, presence_penalty=0.2,
    )
    assert sent["temperature"] == 0.5
    assert sent["p"] == 0.8
    assert sent["k"] == 10
    assert sent["max_tokens"] == 100
    assert sent["stop_sequences"] == ["END"]
    assert sent["seed"] == 42
    assert sent["frequency_penalty"] == 0.1
    assert sent["presence_penalty"] == 0.2


@patch("requests.Session.post")
def test_num_generations_is_dropped_with_debug_log(mock_post):
    from loguru import logger as loguru_logger

    messages: list[str] = []
    sink_id = loguru_logger.add(messages.append, level="DEBUG", format="{message}")
    try:
        sent = _call_cohere(
            mock_post, [{"role": "user", "content": "hi"}],
            n=3,
        )
    finally:
        loguru_logger.remove(sink_id)

    assert "num_generations" not in sent
    assert any("num_generations" in m for m in messages)


@patch("requests.Session.post")
def test_plain_chat_payload_unchanged(mock_post):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]
    sent = _call_cohere(mock_post, messages)

    assert "tools" not in sent
    assert sent["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]


# ---------------------------------------------------------------------------
# Task 2: request-side tools + tool history
# ---------------------------------------------------------------------------

OPENAI_TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "Evaluate math.",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}]


@patch("requests.Session.post")
def test_openai_tools_passthrough_into_v2_payload(mock_post):
    """v2 IS OpenAI-shaped end-to-end -- tools pass through, normalized
    onto the canonical {"type":"function","function":{...}} shape."""
    sent = _call_cohere(
        mock_post, [{"role": "user", "content": "2+2?"}],
        tools=OPENAI_TOOLS,
    )
    assert sent["tools"] == [{
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate math.",
            "parameters": OPENAI_TOOLS[0]["function"]["parameters"],
        },
    }]


@patch("requests.Session.post")
def test_blank_name_tool_is_dropped_locally(mock_post):
    """Mirrors `_google_tools_payload`'s blank-name guard: an entry missing
    a usable function name must be dropped rather than forwarded."""
    bad_tools = [{"type": "function", "function": {"name": "  "}}, OPENAI_TOOLS[0]]
    sent = _call_cohere(
        mock_post, [{"role": "user", "content": "hi"}],
        tools=bad_tools,
    )
    assert [t["function"]["name"] for t in sent["tools"]] == ["calculator"]


@patch("requests.Session.post")
def test_assistant_tool_calls_history_converts_to_v2_shape(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "call_A", "type": "function",
              "function": {"name": "calculator",
                           "arguments": "{\"expression\": \"2+2\"}"}},
             {"id": "call_B", "type": "function",
              "function": {"name": "calculator",
                           "arguments": "{\"expression\": \"3+3\"}"}}]},
        {"role": "tool", "tool_call_id": "call_A", "content": "4"},
        {"role": "tool", "tool_call_id": "call_B", "content": "6"},
    ]
    sent = _call_cohere(mock_post, messages)["messages"]

    assert sent[1]["role"] == "assistant"
    assert sent[1]["tool_calls"] == [
        {"id": "call_A", "type": "function",
         "function": {"name": "calculator", "arguments": "{\"expression\": \"2+2\"}"}},
        {"id": "call_B", "type": "function",
         "function": {"name": "calculator", "arguments": "{\"expression\": \"3+3\"}"}},
    ]
    assert sent[2] == {"role": "tool", "tool_call_id": "call_A",
                       "content": [{"type": "document", "document": {"data": "4"}}]}
    assert sent[3] == {"role": "tool", "tool_call_id": "call_B",
                       "content": [{"type": "document", "document": {"data": "6"}}]}


@patch("requests.Session.post")
def test_empty_streamed_arguments_echo_as_empty_json_object(mock_post):
    """Live-gate case B finding (2026-07-17): a NO-ARG streamed call
    accumulates arguments="" (tool-call-start seeds "", no deltas follow);
    Cohere 400s the echo ('tool arguments must be a stringified JSON
    object') unless it is normalized to "{}"."""
    messages = [
        {"role": "user", "content": "date?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "get_current_datetime",
                                      "arguments": ""}}]},
        {"role": "tool", "tool_call_id": "call_A", "content": "2026-07-17"},
    ]
    sent = _call_cohere(mock_post, messages)["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == "{}"


@patch("requests.Session.post")
def test_dict_arguments_normalized_via_json_dumps(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": {"expression": "2+2"}}}]},
    ]
    sent = _call_cohere(mock_post, messages)["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == json.dumps({"expression": "2+2"})


@patch("requests.Session.post")
def test_unparseable_string_arguments_pass_through_as_is(mock_post):
    """v2 takes `arguments` as a string regardless -- an unparseable string
    is NOT re-validated/rejected, just forwarded as-is."""
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "calculator", "arguments": "{broken"}}]},
    ]
    sent = _call_cohere(mock_post, messages)["messages"]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == "{broken"


@patch("requests.Session.post")
def test_tool_plan_reattached_from_cohere_tool_plan_extra(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "visible text",
         "cohere_tool_plan": "I should call the calculator.",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "calculator", "arguments": "{}"}}]},
    ]
    sent = _call_cohere(mock_post, messages)["messages"]
    assert sent[1]["tool_plan"] == "I should call the calculator."


@patch("requests.Session.post")
def test_tool_plan_falls_back_to_content_when_no_extra(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "calculator", "arguments": "{}"}}]},
    ]
    sent = _call_cohere(mock_post, messages)["messages"]
    assert sent[1]["tool_plan"] == "Let me check."


@patch("requests.Session.post")
def test_tool_result_missing_id_falls_back_to_most_recent_assistant_id(mock_post):
    """Mirrors google's positional fallback (task-266): a tool-result
    message missing tool_call_id pairs with the most recent assistant
    tool_call id."""
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_X", "type": "function",
                         "function": {"name": "calculator", "arguments": "{}"}}]},
        {"role": "tool", "content": "4"},  # tool_call_id omitted
    ]
    sent = _call_cohere(mock_post, messages)["messages"]
    assert sent[2] == {"role": "tool", "tool_call_id": "call_X",
                       "content": [{"type": "document", "document": {"data": "4"}}]}


@patch("requests.Session.post")
def test_all_junk_tool_calls_fall_back_to_plain_content(mock_post):
    """When every tool_calls entry is junk, the assistant turn must be sent
    as a normal plain-text turn instead of a tool_calls-less message with an
    empty [] array (task-263/task-266 precedent)."""
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "hello",
         "tool_calls": ["junk", {"function": "junk"}, {"function": {"name": ""}}]},
    ]
    sent = _call_cohere(mock_post, messages)["messages"]
    assert sent[1] == {"role": "assistant", "content": "hello"}


# ---------------------------------------------------------------------------
# Task 3: non-streaming response tool_calls
# ---------------------------------------------------------------------------

def _cohere_tool_call_response(text=None, tool_plan=None,
                               finish_reason="TOOL_CALL"):
    message = {"role": "assistant", "content": []}
    if text is not None:
        message["content"] = [{"type": "text", "text": text}]
    message["tool_calls"] = [{
        "id": "call_A", "type": "function",
        "function": {"name": "calculator",
                     "arguments": json.dumps({"expression": "2+2"})},
    }]
    if tool_plan is not None:
        message["tool_plan"] = tool_plan
    return {"id": "resp_2", "message": message, "finish_reason": finish_reason}


@patch("requests.Session.post")
def test_tool_calls_response_normalizes_to_openai_shape(mock_post):
    result = _call_cohere_get_result(
        mock_post, _cohere_tool_call_response(),
        [{"role": "user", "content": "2+2?"}],
    )
    choice = result["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["tool_calls"] == [{
        "id": "call_A", "type": "function",
        "function": {"name": "calculator",
                     "arguments": json.dumps({"expression": "2+2"})},
    }]
    # round-trips through the runtime parser:
    from tldw_chatbook.Agents.native_tools import parse_native_tool_calls
    parsed = parse_native_tool_calls(choice["message"])
    assert parsed[0].name == "calculator"


@patch("requests.Session.post")
def test_tool_calls_absent_means_no_tool_calls_key(mock_post):
    result = _call_cohere_get_result(
        mock_post, _cohere_text_response("hi there"),
        [{"role": "user", "content": "hi"}],
    )
    assert "tool_calls" not in result["choices"][0]["message"]


@patch("requests.Session.post")
def test_tool_plan_preserved_as_cohere_tool_plan_extra(mock_post):
    result = _call_cohere_get_result(
        mock_post,
        _cohere_tool_call_response(tool_plan="I will use the calculator."),
        [{"role": "user", "content": "2+2?"}],
    )
    assert result["choices"][0]["message"]["cohere_tool_plan"] == "I will use the calculator."


@patch("requests.Session.post")
def test_tool_plan_absent_omits_extra(mock_post):
    result = _call_cohere_get_result(
        mock_post, _cohere_tool_call_response(),
        [{"role": "user", "content": "2+2?"}],
    )
    assert "cohere_tool_plan" not in result["choices"][0]["message"]


@patch("requests.Session.post")
def test_text_and_tool_calls_both_present_populate_both(mock_post):
    result = _call_cohere_get_result(
        mock_post, _cohere_tool_call_response(text="Checking."),
        [{"role": "user", "content": "2+2?"}],
    )
    message = result["choices"][0]["message"]
    assert message["content"] == "Checking."
    assert message["tool_calls"][0]["function"]["name"] == "calculator"


@patch("requests.Session.post")
def test_malformed_tool_call_arguments_guaranteed_string(mock_post):
    """Response-side junk: a tool_calls entry whose arguments came through
    as something other than a string (e.g. None) must not crash the parser
    and must guarantee a string on the OpenAI-shape entry."""
    response = {
        "id": "resp_3",
        "message": {"role": "assistant", "content": [],
                    "tool_calls": [{"id": "call_N", "type": "function",
                                    "function": {"name": "calculator",
                                                 "arguments": None}}]},
        "finish_reason": "TOOL_CALL",
    }
    result = _call_cohere_get_result(
        mock_post, response, [{"role": "user", "content": "go"}])
    entry = result["choices"][0]["message"]["tool_calls"][0]
    assert entry["function"]["arguments"] == "{}"


# ---------------------------------------------------------------------------
# Task 4: streaming tool-call fragments + tool_plan extra
#
# NOTE: the exact wire shape of Cohere v2's tool-call-start/-delta stream
# events (and precisely where the "index" field lives) is scout knowledge
# per the plan, not independently verified against the live API in this
# offline task. These fixtures encode the plan's stated shape
# (`delta.message.tool_calls` carrying id/name on start, incremental
# `function.arguments` on delta; a sibling top-level `index`) -- Task 6's
# live gate is authoritative if the real API differs.
# ---------------------------------------------------------------------------

def _cohere_tool_call_start(index, call_id, name):
    return {"type": "tool-call-start", "index": index,
            "delta": {"message": {"tool_calls": {
                "id": call_id, "type": "function",
                "function": {"name": name, "arguments": ""}}}}}


def _cohere_tool_call_delta(index, arguments_piece):
    return {"type": "tool-call-delta", "index": index,
            "delta": {"message": {"tool_calls": {
                "function": {"arguments": arguments_piece}}}}}


def _cohere_tool_plan_delta(text):
    return {"type": "tool-plan-delta", "delta": {"message": {"tool_plan": text}}}


def _tool_call_fragments(chunks):
    return [f for c in chunks
            for f in c["choices"][0].get("delta", {}).get("tool_calls", [])]


@patch("requests.Session.post")
def test_streaming_tool_call_start_emits_first_fragment(mock_post):
    events = [
        _cohere_tool_call_start(0, "call_A", "calculator"),
        {"type": "tool-call-end", "index": 0},
        {"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "2+2?"}])
    chunks = _decode_sse_chunks(raw)
    fragments = _tool_call_fragments(chunks)
    assert fragments == [{"index": 0, "id": "call_A", "type": "function",
                          "function": {"name": "calculator", "arguments": ""}}]


@patch("requests.Session.post")
def test_streaming_tool_call_delta_appends_arguments_substring(mock_post):
    events = [
        _cohere_tool_call_start(0, "call_A", "calculator"),
        _cohere_tool_call_delta(0, '{"expres'),
        _cohere_tool_call_delta(0, 'sion": "2+2"}'),
        {"type": "tool-call-end", "index": 0},
        {"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "2+2?"}])
    chunks = _decode_sse_chunks(raw)
    fragments = _tool_call_fragments(chunks)
    assert fragments[1] == {"index": 0, "function": {"arguments": '{"expres'}}
    assert fragments[2] == {"index": 0, "function": {"arguments": 'sion": "2+2"}'}}


@patch("requests.Session.post")
def test_streaming_tool_plan_accumulated_and_emitted_on_first_fragment(mock_post):
    events = [
        _cohere_tool_plan_delta("I should "),
        _cohere_tool_plan_delta("use the calculator."),
        _cohere_tool_call_start(0, "call_A", "calculator"),
        _cohere_tool_call_delta(0, "{}"),
        {"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "2+2?"}])
    chunks = _decode_sse_chunks(raw)
    fragments = _tool_call_fragments(chunks)
    assert fragments[0]["cohere_tool_plan"] == "I should use the calculator."
    # continuation fragments must NOT re-carry the plan text:
    assert "cohere_tool_plan" not in fragments[1]


@patch("requests.Session.post")
def test_streaming_message_end_after_tool_calls_finish_reason_tool_calls(mock_post):
    events = [
        _cohere_tool_call_start(0, "call_A", "calculator"),
        _cohere_tool_call_delta(0, "{}"),
        {"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "2+2?"}])
    chunks = _decode_sse_chunks(raw)
    finishes = [c["choices"][0].get("finish_reason") for c in chunks
                if c["choices"][0].get("finish_reason")]
    assert finishes == ["tool_calls"]


@patch("requests.Session.post")
def test_streaming_two_tool_calls_get_distinct_indexes(mock_post):
    events = [
        _cohere_tool_call_start(0, "call_A", "calculator"),
        _cohere_tool_call_delta(0, "{}"),
        {"type": "tool-call-end", "index": 0},
        _cohere_tool_call_start(1, "call_B", "get_current_datetime"),
        _cohere_tool_call_delta(1, "{}"),
        {"type": "tool-call-end", "index": 1},
        {"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "go"}])
    chunks = _decode_sse_chunks(raw)
    fragments = _tool_call_fragments(chunks)
    starts = [f for f in fragments if "id" in f]
    assert [(f["index"], f["id"], f["function"]["name"]) for f in starts] == [
        (0, "call_A", "calculator"), (1, "call_B", "get_current_datetime")]


@patch("requests.Session.post")
def test_streaming_position_fallback_when_index_missing(mock_post):
    """When the stream event omits an explicit index entirely, positions
    must still be assigned via a running 0-based counter rather than
    crashing or colliding."""
    events = [
        {"type": "tool-call-start", "delta": {"message": {"tool_calls": {
            "id": "call_A", "type": "function",
            "function": {"name": "calculator", "arguments": ""}}}}},
        {"type": "tool-call-start", "delta": {"message": {"tool_calls": {
            "id": "call_B", "type": "function",
            "function": {"name": "get_current_datetime", "arguments": ""}}}}},
        {"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}},
    ]
    raw = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                              [{"role": "user", "content": "go"}])
    chunks = _decode_sse_chunks(raw)
    fragments = _tool_call_fragments(chunks)
    assert [f["index"] for f in fragments] == [0, 1]


@patch("requests.Session.post")
def test_streaming_fragments_reassemble_via_gateway_accumulator(mock_post):
    """Cross-layer contract pin: feed this handler's yielded SSE strings
    through the REAL gateway accumulator path (`_decode_stream_item` +
    `_ToolCallAccumulator` from `tldw_chatbook.Chat.console_provider_gateway`,
    task-243) and assert they reassemble into one merged tool call with the
    `cohere_tool_plan` extra preserved (mirrors task-266's
    `google_thought_signature` cross-layer pin)."""
    from tldw_chatbook.Chat.console_provider_gateway import (
        _decode_stream_item, _ToolCallAccumulator,
    )

    events = [
        _cohere_tool_plan_delta("I should use the calculator."),
        _cohere_tool_call_start(0, "call_A", "calculator"),
        _cohere_tool_call_delta(0, '{"expres'),
        _cohere_tool_call_delta(0, 'sion": "2+2"}'),
        {"type": "tool-call-end", "index": 0},
        {"type": "message-end", "delta": {"finish_reason": "TOOL_CALL"}},
    ]
    sse_lines = _call_cohere_stream(mock_post, _cohere_stream_lines(events),
                                    [{"role": "user", "content": "2+2?"}])

    accumulator = _ToolCallAccumulator()
    for line in sse_lines:
        accumulator.feed_payload(_decode_stream_item(line))

    calls = accumulator.calls()
    assert len(calls) == 1
    assert calls[0]["id"] == "call_A"
    assert calls[0]["function"]["name"] == "calculator"
    assert calls[0]["function"]["arguments"] == json.dumps({"expression": "2+2"})
    assert calls[0]["cohere_tool_plan"] == "I should use the calculator."
