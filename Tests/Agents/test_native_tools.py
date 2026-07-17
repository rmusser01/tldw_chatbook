# Tests/Agents/test_native_tools.py
"""native_tools: capability set, OpenAI conversion, response parsing."""
import json

from tldw_chatbook.Agents.agent_models import ToolSchema
from tldw_chatbook.Agents.native_tools import (
    NATIVE_TOOLS_PROVIDERS, parse_native_tool_calls,
    provider_supports_native_tools, schemas_to_openai_tools,
)
from tldw_chatbook.Chat.Chat_Functions import PROVIDER_PARAM_MAP


def test_capability_set_membership():
    assert provider_supports_native_tools("openai")
    assert provider_supports_native_tools("groq")
    assert provider_supports_native_tools("OpenAI")   # case-insensitive
    assert not provider_supports_native_tools("llama_cpp")
    assert not provider_supports_native_tools("local_llamacpp")
    assert not provider_supports_native_tools("anthropic")  # normalizer drops tool_use
    assert not provider_supports_native_tools("")
    assert not provider_supports_native_tools(None)


def test_every_native_provider_forwards_tools_in_param_map():
    for provider in NATIVE_TOOLS_PROVIDERS:
        mapping = PROVIDER_PARAM_MAP.get(provider)
        assert mapping is not None, provider
        assert mapping.get("tools") == "tools", provider


def test_schemas_to_openai_tools_shape_and_empty_parameters_default():
    schema = ToolSchema(id="b:calc", name="calculator",
                        description="Evaluate math.",
                        parameters={"type": "object",
                                    "properties": {"expression": {"type": "string"}},
                                    "required": ["expression"]})
    bare = ToolSchema(id="b:ping", name="ping", description="Ping.", parameters={})
    tools = schemas_to_openai_tools([schema, bare])
    assert tools[0] == {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate math.",
        "parameters": schema.parameters}}
    assert tools[1]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert schemas_to_openai_tools([]) == []


def _raw_call(name, args, call_id="c1"):
    return {"id": call_id, "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)}}


def test_parse_native_tool_calls_happy_path_and_order():
    message = {"content": None, "tool_calls": [
        _raw_call("calculator", {"expression": "2+2"}, "a"),
        _raw_call("get_current_datetime", {}, "b")]}
    calls = parse_native_tool_calls(message)
    assert [(c.name, c.args, c.call_id) for c in calls] == [
        ("calculator", {"expression": "2+2"}, "a"),
        ("get_current_datetime", {}, "b")]


def test_parse_native_tool_calls_malformed_and_junk():
    message = {"tool_calls": [
        {"id": "x", "type": "function",
         "function": {"name": "calculator", "arguments": "{not json"}},
        {"id": "y", "type": "function",
         "function": {"name": "calculator", "arguments": {"expression": "1"}}},
        {"id": "z", "type": "function", "function": {"name": ""}},
        "junk", {"function": "junk"}]}
    calls = parse_native_tool_calls(message)
    # Malformed arguments -> args={} (the tool's own validation error is
    # echoed back so the model can retry); dict arguments accepted as-is;
    # nameless/junk entries dropped.
    assert [(c.name, c.args) for c in calls] == [
        ("calculator", {}), ("calculator", {"expression": "1"})]
    assert parse_native_tool_calls({}) == ()
    assert parse_native_tool_calls({"tool_calls": None}) == ()
    assert parse_native_tool_calls(None) == ()


def test_ensure_tool_call_ids_synthesizes_missing_ids_only():
    from tldw_chatbook.Agents.native_tools import ensure_tool_call_ids

    raw = [
        {"type": "function", "function": {"name": "calculator", "arguments": "{}"}},
        {"id": "keep-me", "type": "function",
         "function": {"name": "ping", "arguments": "{}"}},
        "junk",
    ]
    normalized = ensure_tool_call_ids(raw)
    assert normalized[0]["id"] == "call_0"
    assert normalized[1]["id"] == "keep-me"
    assert normalized[2] == "junk"
    assert raw[0].get("id") is None      # input entries never mutated
    assert ensure_tool_call_ids(None) == []
    assert ensure_tool_call_ids([]) == []
