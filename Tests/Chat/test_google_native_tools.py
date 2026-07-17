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
