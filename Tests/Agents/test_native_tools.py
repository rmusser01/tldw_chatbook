# Tests/Agents/test_native_tools.py
"""native_tools: capability set, OpenAI conversion, response parsing."""

import json

from tldw_chatbook.Agents.agent_models import ToolSchema
from tldw_chatbook.Agents.native_tools import (
    NATIVE_TOOLS_PROVIDERS,
    parse_native_tool_calls,
    provider_supports_native_tools,
    schemas_to_openai_tools,
)
from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP
from tldw_chatbook.LLM_Calls.qwencloud import (
    build_qwencloud_payload,
    normalize_qwencloud_response,
)


def test_capability_set_membership():
    assert provider_supports_native_tools("openai")
    assert provider_supports_native_tools("groq")
    assert provider_supports_native_tools("OpenAI")  # case-insensitive
    assert not provider_supports_native_tools("llama_cpp")
    assert not provider_supports_native_tools("local_llamacpp")
    assert provider_supports_native_tools("anthropic")  # converted (task-263)
    assert provider_supports_native_tools("google")  # converted (task-266)
    assert provider_supports_native_tools("cohere")  # converted via v2 /chat (task-267)
    assert not provider_supports_native_tools("")
    assert not provider_supports_native_tools(None)


def test_every_native_provider_forwards_tools_in_param_map():
    for provider in NATIVE_TOOLS_PROVIDERS:
        mapping = PROVIDER_PARAM_MAP.get(provider)
        assert mapping is not None, provider
        assert mapping.get("tools") == "tools", provider


def test_zai_native_provider_contract_is_eligible() -> None:
    assert "zai" in NATIVE_TOOLS_PROVIDERS
    assert PROVIDER_PARAM_MAP["zai"]["tools"] == "tools"


def test_native_provider_contract_requires_qwencloud_dispatch_and_history():
    """QwenCloud may be native only after all three seam invariants hold."""
    assert "qwencloud" in NATIVE_TOOLS_PROVIDERS
    assert PROVIDER_PARAM_MAP["qwencloud"]["tools"] == "tools"

    normalized = normalize_qwencloud_response(
        {
            "status": "completed",
            "output": [
                {
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_A",
                    "name": "calculator",
                    "arguments": '{"expression":"6*7"}',
                },
                {
                    "type": "function_call",
                    "status": "completed",
                    "call_id": "call_B",
                    "name": "calculator",
                    "arguments": '{"expression":"8*8"}',
                },
            ],
        },
        api_mode="responses",
    )
    assert normalized["choices"][0]["message"]["tool_calls"] == [
        {
            "id": "call_A",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"6*7"}',
            },
        },
        {
            "id": "call_B",
            "type": "function",
            "function": {
                "name": "calculator",
                "arguments": '{"expression":"8*8"}',
            },
        },
    ]

    continuation = [
        {"role": "user", "content": "Calculate it."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": normalized["choices"][0]["message"]["tool_calls"],
        },
        # Canonical runtime history is accepted even when result rows arrive
        # out of call order; Responses must restore call/output adjacency.
        {"role": "tool", "tool_call_id": "call_B", "content": "64"},
        {"role": "tool", "tool_call_id": "call_A", "content": "42"},
    ]
    responses_payload = build_qwencloud_payload(
        api_mode="responses",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=continuation,
        streaming=False,
    )
    assert responses_payload["input"][-4:] == [
        {
            "type": "function_call",
            "call_id": "call_A",
            "name": "calculator",
            "arguments": '{"expression":"6*7"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_A",
            "output": "42",
        },
        {
            "type": "function_call",
            "call_id": "call_B",
            "name": "calculator",
            "arguments": '{"expression":"8*8"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call_B",
            "output": "64",
        },
    ]
    chat_payload = build_qwencloud_payload(
        api_mode="chat_completions",
        model="qwen3.8-max",
        system_message=None,
        messages_payload=continuation,
        streaming=False,
    )
    assert chat_payload["messages"] == continuation


def test_schemas_to_openai_tools_shape_and_empty_parameters_default():
    schema = ToolSchema(
        id="b:calc",
        name="calculator",
        description="Evaluate math.",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    )
    bare = ToolSchema(id="b:ping", name="ping", description="Ping.", parameters={})
    tools = schemas_to_openai_tools([schema, bare])
    assert tools[0] == {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Evaluate math.",
            "parameters": schema.parameters,
        },
    }
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert schemas_to_openai_tools([]) == []


def _raw_call(name, args, call_id="c1"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_parse_native_tool_calls_happy_path_and_order():
    message = {
        "content": None,
        "tool_calls": [
            _raw_call("calculator", {"expression": "2+2"}, "a"),
            _raw_call("get_current_datetime", {}, "b"),
        ],
    }
    calls = parse_native_tool_calls(message)
    assert [(c.name, c.args, c.call_id) for c in calls] == [
        ("calculator", {"expression": "2+2"}, "a"),
        ("get_current_datetime", {}, "b"),
    ]


def test_parse_native_tool_calls_preserves_exact_string_arguments() -> None:
    raw_arguments = '{ "b": 2, "a": 1 }'
    calls = parse_native_tool_calls(
        {
            "tool_calls": [
                {
                    "id": "exact",
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "arguments": raw_arguments,
                    },
                }
            ]
        }
    )

    assert calls[0].args == {"a": 1, "b": 2}
    assert calls[0].raw_arguments == raw_arguments


def test_parse_native_tool_calls_malformed_and_junk():
    message = {
        "tool_calls": [
            {
                "id": "x",
                "type": "function",
                "function": {"name": "calculator", "arguments": "{not json"},
            },
            {
                "id": "y",
                "type": "function",
                "function": {"name": "calculator", "arguments": {"expression": "1"}},
            },
            {"id": "z", "type": "function", "function": {"name": ""}},
            "junk",
            {"function": "junk"},
        ]
    }
    calls = parse_native_tool_calls(message)
    # Malformed arguments -> args={} (the tool's own validation error is
    # echoed back so the model can retry); dict arguments accepted as-is;
    # nameless/junk entries dropped.
    assert [(c.name, c.args) for c in calls] == [
        ("calculator", {}),
        ("calculator", {"expression": "1"}),
    ]
    assert [c.raw_arguments for c in calls] == ["{not json", ""]
    assert parse_native_tool_calls({}) == ()
    assert parse_native_tool_calls({"tool_calls": None}) == ()
    assert parse_native_tool_calls(None) == ()


def test_ensure_tool_call_ids_synthesizes_missing_ids_only():
    from tldw_chatbook.Agents.native_tools import ensure_tool_call_ids

    raw = [
        {"type": "function", "function": {"name": "calculator", "arguments": "{}"}},
        {
            "id": "keep-me",
            "type": "function",
            "function": {"name": "ping", "arguments": "{}"},
        },
        "junk",
    ]
    normalized = ensure_tool_call_ids(raw)
    assert normalized[0]["id"] == "call_0"
    assert normalized[1]["id"] == "keep-me"
    assert normalized[2] == "junk"
    assert raw[0].get("id") is None  # input entries never mutated
    assert ensure_tool_call_ids(None) == []
    assert ensure_tool_call_ids([]) == []
