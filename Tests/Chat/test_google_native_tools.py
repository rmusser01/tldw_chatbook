"""
Tests for Google/Gemini native tool-calls request-side conversion (task-266
Task 1).

`chat_with_google` must convert OpenAI-shaped ``tools`` into Gemini's native
``functionDeclarations`` wrapping, and must convert OpenAI-shaped tool-call
history (assistant ``tool_calls`` + ``role="tool"`` messages) into Gemini's
``functionCall`` / ``functionResponse`` parts before sending the request.

Mirrors the mocking pattern from `Tests/Chat/test_anthropic_native_tools.py`:
patch `requests.Session.post`, drive the real dispatcher via `chat_api_call`,
and inspect the JSON payload actually sent.
"""

import json
from unittest.mock import Mock, patch

from tldw_chatbook.Chat.Chat_Functions import chat_api_call


def _gemini_text_response(text="ok"):
    return {"candidates": [{"content": {"parts": [{"text": text}],
                                        "role": "model"},
                            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1}}


OPENAI_TOOLS = [{"type": "function", "function": {
    "name": "calculator", "description": "Evaluate math.",
    "parameters": {"type": "object",
                   "properties": {"expression": {"type": "string"}},
                   "required": ["expression"]}}}]


def _call_google(mock_post, messages, **extra):
    """Shared mock plumbing: stub a non-streaming Gemini response and drive
    the real dispatch path, returning the JSON body actually posted."""
    mock_response = Mock()
    mock_response.json.return_value = _gemini_text_response()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    chat_api_call(
        "google",
        messages_payload=messages,
        api_key="test-key",
        model="gemini-1.5-flash-latest",
        streaming=False,
        **extra,
    )
    return mock_post.call_args[1]["json"]


@patch("requests.Session.post")
def test_openai_tools_wrap_as_function_declarations(mock_post):
    sent = _call_google(
        mock_post,
        [{"role": "user", "content": "2+2?"}],
        tools=OPENAI_TOOLS,
    )
    assert sent["tools"] == [{"functionDeclarations": [{
        "name": "calculator", "description": "Evaluate math.",
        "parameters": OPENAI_TOOLS[0]["function"]["parameters"]}]}]


@patch("requests.Session.post")
def test_gemini_shaped_tools_pass_through_untouched(mock_post):
    native = [{"functionDeclarations": [{"name": "t", "parameters": {}}]}]
    sent = _call_google(
        mock_post,
        [{"role": "user", "content": "hi"}],
        tools=native,
    )
    assert sent["tools"] == native


@patch("requests.Session.post")
def test_blank_name_openai_tool_dropped_locally(mock_post):
    """Gemini rejects empty tool names — an OpenAI tool entry with a blank
    name must be dropped locally (task-263 review precedent)."""
    bad_tools = [{"type": "function", "function": {"name": "  "}}, OPENAI_TOOLS[0]]
    sent = _call_google(
        mock_post,
        [{"role": "user", "content": "hi"}],
        tools=bad_tools,
    )
    assert sent["tools"] == [{"functionDeclarations": [{
        "name": "calculator", "description": "Evaluate math.",
        "parameters": OPENAI_TOOLS[0]["function"]["parameters"]}]}]


@patch("requests.Session.post")
def test_openai_tool_history_converts_to_gemini_parts(mock_post):
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
    sent = _call_google(mock_post, messages)
    contents = sent["contents"]

    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [
        {"functionCall": {"name": "calculator", "args": {"expression": "2+2"}}},
        {"functionCall": {"name": "calculator", "args": {"expression": "3+3"}}}]
    # BOTH results coalesce into ONE user turn, positionally ordered
    # (Gemini pairs same-name parallel calls by order):
    assert contents[2]["role"] == "user"
    assert contents[2]["parts"] == [
        {"functionResponse": {"name": "calculator", "response": {"result": "4"}}},
        {"functionResponse": {"name": "calculator", "response": {"result": "6"}}}]
    assert len(contents) == 3


@patch("requests.Session.post")
def test_tool_result_with_json_object_content_passes_object(mock_post):
    """Dict-parseable tool-result content is used directly as the
    functionResponse ``response`` object; non-dict JSON (e.g. a bare array)
    and plain text both wrap as ``{"result": <string>}``."""
    messages = [
        {"role": "user", "content": "Look up three things."},
        {"role": "assistant", "content": "",
         "tool_calls": [
             {"id": "call_A", "type": "function",
              "function": {"name": "lookup", "arguments": "{}"}},
             {"id": "call_B", "type": "function",
              "function": {"name": "lookup", "arguments": "{}"}},
             {"id": "call_C", "type": "function",
              "function": {"name": "lookup", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_A", "content": '{"answer": 42}'},
        {"role": "tool", "tool_call_id": "call_B", "content": "[1, 2]"},
        {"role": "tool", "tool_call_id": "call_C", "content": "plain text"},
    ]
    sent = _call_google(mock_post, messages)
    contents = sent["contents"]

    assert contents[2]["role"] == "user"
    assert contents[2]["parts"] == [
        {"functionResponse": {"name": "lookup", "response": {"answer": 42}}},
        {"functionResponse": {"name": "lookup", "response": {"result": "[1, 2]"}}},
        {"functionResponse": {"name": "lookup", "response": {"result": "plain text"}}}]


@patch("requests.Session.post")
def test_assistant_text_plus_tool_calls_keeps_text_part_first(mock_post):
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "Let me check.",
         "tool_calls": [{"id": "call_1", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{\"expression\": \"2+2\"}"}}]},
    ]
    sent = _call_google(mock_post, messages)
    contents = sent["contents"]

    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [
        {"text": "Let me check."},
        {"functionCall": {"name": "calculator", "args": {"expression": "2+2"}}}]


@patch("requests.Session.post")
def test_all_junk_tool_calls_fall_back_to_plain_content(mock_post):
    """When every tool_calls entry is junk, the assistant turn must be sent
    as a normal plain-text model turn instead of an empty-parts turn that
    gets silently dropped (task-263 precedent)."""
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "hello",
         "tool_calls": ["junk", {"function": "junk"}, {"function": {"name": ""}}]},
    ]
    sent = _call_google(mock_post, messages)
    contents = sent["contents"]

    assert contents[1]["role"] == "model"
    assert contents[1]["parts"] == [{"text": "hello"}]


@patch("requests.Session.post")
def test_unknown_tool_call_id_uses_positional_fallback(mock_post):
    """Gemini has no call ids of its own; when the OpenAI tool_call_id isn't
    found in the accumulated id->name map, pair the nth consecutive tool
    result with the nth functionCall of the preceding model turn."""
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_A", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{\"expression\": \"2+2\"}"}}]},
        {"role": "tool", "tool_call_id": "mystery", "content": "4"},
    ]
    sent = _call_google(mock_post, messages)
    contents = sent["contents"]

    assert contents[2]["role"] == "user"
    assert contents[2]["parts"] == [
        {"functionResponse": {"name": "calculator", "response": {"result": "4"}}}]


@patch("requests.Session.post")
def test_plain_chat_payload_unchanged(mock_post):
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "there"},
    ]
    sent = _call_google(mock_post, messages)

    assert "tools" not in sent
    assert sent["contents"] == [
        {"role": "user", "parts": [{"text": "hi"}]},
        {"role": "model", "parts": [{"text": "there"}]},
    ]


@patch("requests.Session.post")
def test_list_content_with_tool_calls_keeps_text_parts(mock_post):
    """T1 review Important: list-form (multimodal) assistant content
    alongside tool_calls must keep its text parts (same bug class as the
    anthropic sibling, PR #659 review)."""
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant",
         "content": [{"type": "text", "text": "Let me check."},
                     {"type": "image_url", "image_url": {"url": "data:x"}}],
         "tool_calls": [{"id": "call_L", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_L", "content": "4"},
    ]
    sent = _call_google(mock_post, messages)
    model_turn = sent["contents"][1]
    assert model_turn["parts"][0] == {"text": "Let me check."}
    assert "functionCall" in model_turn["parts"][-1]


# ---------------------------------------------------------------------------
# Task 2: response-side normalization (non-streaming pin + streaming emission)
# ---------------------------------------------------------------------------

def _gemini_function_call_response():
    return {"candidates": [{"content": {"parts": [
                {"text": "Checking."},
                {"functionCall": {"name": "calculator",
                                  "args": {"expression": "2+2"}}}],
                "role": "model"},
            "finishReason": "STOP", "index": 0}],
            "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 9}}


def _call_google_get_result(mock_post, response_json, messages, **extra):
    """Like `_call_google` but returns the normalized `chat_api_call` result
    instead of the request JSON that was sent."""
    mock_response = Mock()
    mock_response.json.return_value = response_json
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_post.return_value = mock_response

    return chat_api_call(
        "google",
        messages_payload=messages,
        api_key="test-key",
        model="gemini-1.5-flash-latest",
        streaming=False,
        **extra,
    )


@patch("requests.Session.post")
def test_non_streaming_function_call_normalizes_to_tool_calls(mock_post):
    """PIN of EXISTING behavior (scout: lines ~1868-1884): functionCall
    parts already normalize to OpenAI tool_calls with synthesized ids."""
    result = _call_google_get_result(
        mock_post, _gemini_function_call_response(),
        [{"role": "user", "content": "2+2?"}],
    )
    message = result["choices"][0]["message"]
    (entry,) = message["tool_calls"]
    assert entry["type"] == "function"
    assert entry["function"]["name"] == "calculator"
    assert json.loads(entry["function"]["arguments"]) == {"expression": "2+2"}
    assert entry["id"]  # synthesized, non-empty
    # round-trips through the runtime parser:
    from tldw_chatbook.Agents.native_tools import parse_native_tool_calls
    parsed = parse_native_tool_calls(message)
    assert parsed[0].name == "calculator"


def _gemini_stream_lines(events):
    return [f"data: {json.dumps(e)}" for e in events]


def _call_google_stream(mock_post, sse_lines, messages, **extra):
    """Streaming counterpart of `_call_google_get_result`: stub a streaming
    Gemini response (`iter_lines` yields decoded SSE text lines, matching
    `chat_with_google`'s `iter_lines(decode_unicode=True)`) and return the
    consumed generator's yielded SSE strings."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    mock_response.iter_lines.return_value = sse_lines
    mock_response.close = Mock()
    mock_post.return_value = mock_response

    result = chat_api_call(
        "google",
        messages_payload=messages,
        api_key="test-key",
        model="gemini-1.5-flash-latest",
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


@patch("requests.Session.post")
def test_streaming_function_call_chunk_emits_whole_openai_fragment(mock_post):
    # SSE lines: a text chunk, then a chunk whose parts carry a complete
    # functionCall, then a STOP finish. Assert exactly one delta.tool_calls
    # fragment: [{"index": 0, "id": <non-empty>, "type": "function",
    #   "function": {"name": "calculator",
    #                "arguments": json.dumps({"expression": "2+2"})}}]
    # (Gemini streams functionCall parts WHOLE — one complete fragment is
    # accumulator-compatible: first fragment carries everything.)
    events = [
        {"candidates": [{"content": {"parts": [{"text": "Checking."}],
                                     "role": "model"}, "index": 0}]},
        {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "calculator",
                              "args": {"expression": "2+2"}}}],
            "role": "model"}, "index": 0}]},
        {"candidates": [{"finishReason": "STOP", "index": 0}]},
    ]
    sse_lines = _call_google_stream(
        mock_post, _gemini_stream_lines(events),
        [{"role": "user", "content": "2+2?"}],
    )
    chunks = _decode_sse_chunks(sse_lines)

    fragments = [c["choices"][0]["delta"]["tool_calls"]
                 for c in chunks if "tool_calls" in c["choices"][0].get("delta", {})]
    assert len(fragments) == 1
    assert len(fragments[0]) == 1
    entry = fragments[0][0]
    assert entry["index"] == 0
    assert entry["id"]
    assert entry["type"] == "function"
    assert entry["function"] == {"name": "calculator",
                                  "arguments": json.dumps({"expression": "2+2"})}

    # text still streams; a tool-calls-only chunk (no text) must still yield.
    texts = [c["choices"][0].get("delta", {}).get("content") for c in chunks]
    assert "Checking." in texts
    finishes = [c["choices"][0].get("finish_reason") for c in chunks]
    assert "stop" in finishes


@patch("requests.Session.post")
def test_streaming_two_function_calls_get_distinct_indexes(mock_post):
    # one chunk carrying two functionCall parts -> fragments with index 0
    # and 1, distinct synthesized ids
    events = [
        {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "calculator",
                              "args": {"expression": "2+2"}}},
            {"functionCall": {"name": "calculator",
                              "args": {"expression": "3+3"}}}],
            "role": "model"}, "index": 0}]},
        {"candidates": [{"finishReason": "STOP", "index": 0}]},
    ]
    sse_lines = _call_google_stream(
        mock_post, _gemini_stream_lines(events),
        [{"role": "user", "content": "go"}],
    )
    chunks = _decode_sse_chunks(sse_lines)

    fragments = [c["choices"][0]["delta"]["tool_calls"]
                 for c in chunks if "tool_calls" in c["choices"][0].get("delta", {})]
    assert len(fragments) == 1
    (only_fragment,) = fragments
    assert [f["index"] for f in only_fragment] == [0, 1]
    assert only_fragment[0]["id"] != only_fragment[1]["id"]
    assert only_fragment[0]["function"]["arguments"] == json.dumps({"expression": "2+2"})
    assert only_fragment[1]["function"]["arguments"] == json.dumps({"expression": "3+3"})


@patch("requests.Session.post")
def test_streaming_fragments_reassemble_via_gateway_accumulator(mock_post):
    """Cross-layer contract pin: feed this handler's yielded SSE strings
    through the REAL gateway accumulator path (`_decode_stream_item` +
    `_ToolCallAccumulator` from `tldw_chatbook.Chat.console_provider_gateway`,
    task-243) and assert they reassemble into one merged tool call."""
    from tldw_chatbook.Chat.console_provider_gateway import (
        _decode_stream_item, _ToolCallAccumulator,
    )

    events = [
        {"candidates": [{"content": {"parts": [{"text": "Checking."}],
                                     "role": "model"}, "index": 0}]},
        {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "calculator",
                              "args": {"expression": "2+2"}}}],
            "role": "model"}, "index": 0}]},
        {"candidates": [{"finishReason": "STOP", "index": 0}]},
    ]
    sse_lines = _call_google_stream(
        mock_post, _gemini_stream_lines(events),
        [{"role": "user", "content": "2+2?"}],
    )

    accumulator = _ToolCallAccumulator()
    for line in sse_lines:
        accumulator.feed_payload(_decode_stream_item(line))

    assert accumulator.calls()[0]["function"]["name"] == "calculator"
    assert accumulator.calls()[0]["function"]["arguments"] == json.dumps({"expression": "2+2"})
    assert len(accumulator.calls()) == 1


@patch("requests.Session.post")
def test_streaming_malformed_function_call_is_skipped_not_fatal(mock_post):
    """T2 review Important: a non-dict functionCall part must be skipped
    without aborting the stream — later text still flows (task-263 sibling
    regression class)."""
    chunks = [
        {"candidates": [{"content": {"parts": [
            {"functionCall": "not-a-dict"}], "role": "model"}, "index": 0}]},
        {"candidates": [{"content": {"parts": [
            {"text": "still alive"}], "role": "model"}, "index": 0}]},
        {"candidates": [{"content": {"parts": []}, "role": "model",
                         "finishReason": "STOP", "index": 0}]},
    ]
    raw = _call_google_stream(mock_post, _gemini_stream_lines(chunks),
                              [{"role": "user", "content": "go"}])
    parsed = _decode_sse_chunks(raw)
    texts = [c["choices"][0].get("delta", {}).get("content") for c in parsed]
    assert "still alive" in texts
    assert not any("error" in c for c in parsed)


@patch("requests.Session.post")
def test_streaming_index_continues_across_chunks_and_blank_names_skip(mock_post):
    """T2 review Minors: the tool position runs across the WHOLE stream
    (chunk 2's call gets index 1), and a blank-name part neither emits nor
    consumes a position."""
    chunks = [
        {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "calculator",
                              "args": {"expression": "2+2"}}}],
            "role": "model"}, "index": 0}]},
        {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "  ", "args": {}}},
            {"functionCall": {"name": "get_current_datetime", "args": {}}}],
            "role": "model"}, "index": 0}]},
        {"candidates": [{"content": {"parts": []}, "role": "model",
                         "finishReason": "STOP", "index": 0}]},
    ]
    raw = _call_google_stream(mock_post, _gemini_stream_lines(chunks),
                              [{"role": "user", "content": "go"}])
    parsed = _decode_sse_chunks(raw)
    fragments = [f for c in parsed
                 for f in c["choices"][0].get("delta", {}).get("tool_calls", [])]
    assert [(f["index"], f["function"]["name"]) for f in fragments] == [
        (0, "calculator"), (1, "get_current_datetime")]


@patch("requests.Session.post")
def test_thought_signature_round_trips_response_to_request(mock_post):
    """task-266 live gate: Gemini 3 models 400 unless the response part's
    thoughtSignature is echoed back verbatim on the follow-up request's
    functionCall part. Non-streaming response carries it opaquely on the
    OpenAI entry; the request converter re-attaches it."""
    response = {"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "calculator",
                          "args": {"expression": "2+2"}},
         "thoughtSignature": "sig-abc"}], "role": "model"},
        "finishReason": "STOP", "index": 0}],
        "usageMetadata": {}}
    result = _call_google_get_result(
        mock_post, response, [{"role": "user", "content": "2+2?"}])
    (entry,) = result["choices"][0]["message"]["tool_calls"]
    assert entry["google_thought_signature"] == "sig-abc"

    # Echo the entry back as history: the functionCall part must carry it.
    messages = [
        {"role": "user", "content": "2+2?"},
        {"role": "assistant", "content": "", "tool_calls": [entry]},
        {"role": "tool", "tool_call_id": entry["id"], "content": "4"},
    ]
    sent = _call_google(mock_post, messages)
    model_part = sent["contents"][1]["parts"][0]
    assert model_part["thoughtSignature"] == "sig-abc"
    assert model_part["functionCall"]["name"] == "calculator"


@patch("requests.Session.post")
def test_streaming_fragment_carries_thought_signature(mock_post):
    """Streaming parity: a streamed functionCall part's thoughtSignature
    rides on the emitted OpenAI fragment."""
    chunks = [
        {"candidates": [{"content": {"parts": [
            {"functionCall": {"name": "calculator",
                              "args": {"expression": "2+2"}},
             "thoughtSignature": "sig-str"}], "role": "model"}, "index": 0}]},
        {"candidates": [{"content": {"parts": []}, "role": "model",
                         "finishReason": "STOP", "index": 0}]},
    ]
    raw = _call_google_stream(mock_post, _gemini_stream_lines(chunks),
                              [{"role": "user", "content": "go"}])
    parsed = _decode_sse_chunks(raw)
    fragments = [f for c in parsed
                 for f in c["choices"][0].get("delta", {}).get("tool_calls", [])]
    assert fragments[0]["google_thought_signature"] == "sig-str"


@patch("requests.Session.post")
def test_non_streaming_malformed_function_call_part_is_skipped(mock_post):
    """PR #662 review (Gemini): a non-dict functionCall in a NON-streaming
    response must be skipped, not crash the parser (mirrors the streaming
    guard)."""
    response = {"candidates": [{"content": {"parts": [
        {"functionCall": "not-a-dict"},
        {"functionCall": {"name": "calculator",
                          "args": {"expression": "2+2"}}}],
        "role": "model"}, "finishReason": "STOP", "index": 0}],
        "usageMetadata": {}}
    result = _call_google_get_result(
        mock_post, response, [{"role": "user", "content": "2+2?"}])
    entries = result["choices"][0]["message"]["tool_calls"]
    assert len(entries) == 1
    assert entries[0]["function"]["name"] == "calculator"


@patch("requests.Session.post")
def test_unpairable_tool_result_is_skipped_not_empty_named(mock_post):
    """PR #662 review (Qodo): a role=tool result whose id misses the map AND
    whose positional fallback is exhausted must be SKIPPED — an empty-name
    functionResponse would 400 the whole request."""
    messages = [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": "",
         "tool_calls": [{"id": "call_X", "type": "function",
                         "function": {"name": "calculator",
                                      "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "call_X", "content": "4"},
        {"role": "tool", "tool_call_id": "mystery-2", "content": "orphan"},
    ]
    sent = _call_google(mock_post, messages)
    result_turn = sent["contents"][2]
    names = [p["functionResponse"]["name"] for p in result_turn["parts"]]
    assert names == ["calculator"]  # the orphan (position 1 > 0 calls) dropped
