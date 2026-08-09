"""Strict registration and raw dispatch tests for Chatbook's MCP gateway adapter."""

from __future__ import annotations

import copy
from typing import Any

import pytest

gateway = pytest.importorskip(
    "mcp_unified.gateway", reason="mcp-unified extra not installed"
)
GatewayRequestContext = gateway.GatewayRequestContext
GatewayToolExecutionError = gateway.GatewayToolExecutionError

from tldw_chatbook.MCP.gateway_runtime import ChatbookGatewayRuntime  # noqa: E402
from tldw_chatbook.MCP.server import (  # noqa: E402
    TldwMCPServer,
    _describe_local_tools,
)


BUILTIN_TOOL_NAMES = [
    "chat_with_llm",
    "chat_with_character",
    "search_rag",
    "search_conversations",
    "create_note",
    "search_notes",
    "list_characters",
    "get_conversation_history",
    "export_conversation",
    "ingest_media",
]


def _context() -> GatewayRequestContext:
    return GatewayRequestContext(request_id="test-request")


def _descriptor(name: str = "echo") -> dict[str, Any]:
    return {
        "name": name,
        "description": f"{name} tool",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    }


def _runtime(*descriptors: dict[str, Any]):
    return ChatbookGatewayRuntime(
        name="tldw_chatbook",
        version="0.1.0",
        tool_descriptors=list(descriptors) or [_descriptor()],
    )


def _register_real_builtins(runtime):
    server = TldwMCPServer.__new__(TldwMCPServer)
    server.mcp = runtime
    server._register_tools()
    return server


def test_runtime_requires_one_handler_for_every_expected_builtin() -> None:
    descriptors = _describe_local_tools()
    runtime = _runtime(*descriptors)

    for descriptor in descriptors[:-1]:

        @runtime.tool(name=descriptor["name"])
        async def handler() -> None:
            return None

    with pytest.raises(ValueError, match="descriptor.*handler|handler.*descriptor"):
        runtime.finalize()


def test_runtime_rejects_duplicate_tool_descriptor_names() -> None:
    descriptor = _descriptor()

    with pytest.raises(ValueError, match="duplicate tool descriptor"):
        _runtime(descriptor, copy.deepcopy(descriptor))


def test_runtime_rejects_duplicate_tool_handler_names() -> None:
    runtime = _runtime(_descriptor())

    @runtime.tool()
    async def echo() -> None:
        return None

    with pytest.raises(ValueError, match="duplicate tool handler"):

        @runtime.tool(name="echo")
        async def duplicate() -> None:
            return None


def test_runtime_rejects_handler_without_descriptor() -> None:
    runtime = _runtime(_descriptor())

    @runtime.tool()
    async def echo() -> None:
        return None

    @runtime.tool()
    async def extra() -> None:
        return None

    with pytest.raises(ValueError, match="descriptor.*handler|handler.*descriptor"):
        runtime.finalize()


def test_runtime_rejects_descriptor_without_handler() -> None:
    runtime = _runtime(_descriptor())

    with pytest.raises(ValueError, match="descriptor.*handler|handler.*descriptor"):
        runtime.finalize()


@pytest.mark.asyncio
async def test_all_ten_builtin_handlers_register_with_exact_names() -> None:
    runtime = _runtime(*_describe_local_tools())
    _register_real_builtins(runtime)
    runtime.finalize()

    descriptors = await runtime.list_tools(_context())

    assert [descriptor["name"] for descriptor in descriptors] == BUILTIN_TOOL_NAMES
    assert list(runtime._tool_handlers) == BUILTIN_TOOL_NAMES
    assert [handler.__name__ for handler in runtime._tool_handlers.values()] == (
        BUILTIN_TOOL_NAMES
    )


def test_all_ten_builtin_schemas_reject_additional_properties() -> None:
    descriptors = _describe_local_tools()

    assert [descriptor["name"] for descriptor in descriptors] == BUILTIN_TOOL_NAMES
    assert all(
        descriptor["inputSchema"]["type"] == "object"
        and descriptor["inputSchema"]["additionalProperties"] is False
        for descriptor in descriptors
    )


def test_runtime_rejects_descriptor_that_allows_additional_properties() -> None:
    descriptor = _describe_local_tools()[0]
    descriptor["inputSchema"]["additionalProperties"] = True

    with pytest.raises(ValueError, match="additionalProperties"):
        _runtime(descriptor)


def test_standalone_tool_descriptors_exclude_library_tools() -> None:
    descriptors = _describe_local_tools()

    assert descriptors
    assert not any(
        descriptor["name"].startswith("library_") for descriptor in descriptors
    )


@pytest.mark.parametrize(
    "application_value",
    [
        pytest.param({"answer": 42}, id="dictionary"),
        pytest.param(["one", "two"], id="list"),
        pytest.param("plain text", id="string"),
    ],
)
@pytest.mark.asyncio
async def test_call_tool_returns_application_values_unchanged(
    application_value: Any,
) -> None:
    runtime = _runtime(_descriptor())

    @runtime.tool()
    async def echo() -> Any:
        return application_value

    runtime.finalize()

    result = await runtime.call_tool("echo", {}, _context())

    assert result is application_value


@pytest.mark.asyncio
async def test_call_tool_keeps_error_dictionary_as_application_data() -> None:
    runtime = _runtime(_descriptor("ordinary_builtin"))
    application_data = {"error": "application data"}

    @runtime.tool()
    async def ordinary_builtin() -> dict[str, str]:
        return application_data

    runtime.finalize()

    result = await runtime.call_tool("ordinary_builtin", {}, _context())

    assert result is application_data


@pytest.mark.asyncio
async def test_call_tool_invokes_existing_handler_with_keyword_arguments() -> None:
    descriptor = _descriptor("join")
    descriptor["inputSchema"]["properties"] = {
        "left": {"type": "string"},
        "right": {"type": "string"},
    }
    descriptor["inputSchema"]["required"] = ["left", "right"]
    runtime = _runtime(descriptor)

    @runtime.tool()
    async def join(left: str, right: str) -> str:
        return left + right

    runtime.finalize()

    assert (
        await runtime.call_tool("join", {"left": "chat", "right": "book"}, _context())
        == "chatbook"
    )


@pytest.mark.asyncio
async def test_call_tool_rejects_unknown_name_with_bounded_public_tool_error() -> None:
    runtime = _runtime(_descriptor())

    @runtime.tool()
    async def echo() -> None:
        return None

    runtime.finalize()

    with pytest.raises(GatewayToolExecutionError) as exc_info:
        await runtime.call_tool("missing", {}, _context())

    assert exc_info.value.kind == "tool"
    assert exc_info.value.reason_code == "tool_not_found"
    assert 0 < len(exc_info.value.public_message) <= 512
    assert "missing" not in exc_info.value.public_message


@pytest.mark.asyncio
async def test_list_tools_returns_defensive_copies_in_descriptor_order() -> None:
    first = _descriptor("first")
    second = _descriptor("second")
    runtime = _runtime(first, second)
    first["description"] = "mutated input"

    @runtime.tool()
    async def first() -> None:
        return None

    @runtime.tool()
    async def second() -> None:
        return None

    runtime.finalize()

    published = await runtime.list_tools(_context())
    published[0]["description"] = "mutated output"
    published[0]["inputSchema"]["properties"]["surprise"] = {"type": "string"}

    listed_again = await runtime.list_tools(_context())
    assert [descriptor["name"] for descriptor in listed_again] == ["first", "second"]
    assert listed_again[0]["description"] == "first tool"
    assert "surprise" not in listed_again[0]["inputSchema"]["properties"]


@pytest.mark.parametrize("field", ["name", "version"])
@pytest.mark.parametrize(
    "invalid_value",
    [
        pytest.param("", id="empty"),
        pytest.param(None, id="none"),
        pytest.param(7, id="integer"),
        pytest.param("x" * 513, id="over-bound"),
    ],
)
def test_runtime_rejects_invalid_public_identity(
    field: str, invalid_value: object
) -> None:
    values = {"name": "tldw_chatbook", "version": "0.1.0"}
    values[field] = invalid_value

    with pytest.raises(ValueError, match="bounded string"):
        ChatbookGatewayRuntime(tool_descriptors=[_descriptor()], **values)


def test_runtime_retains_bounded_public_identity() -> None:
    runtime = ChatbookGatewayRuntime(
        name="n" * 512,
        version="v" * 512,
        tool_descriptors=[_descriptor()],
    )

    assert runtime.name == "n" * 512
    assert runtime.version == "v" * 512


@pytest.mark.asyncio
async def test_runtime_does_not_serve_or_register_after_finalization() -> None:
    runtime = _runtime(_descriptor())

    with pytest.raises(RuntimeError, match="finalized"):
        await runtime.list_tools(_context())

    @runtime.tool()
    async def echo() -> None:
        return None

    runtime.finalize()

    with pytest.raises(RuntimeError, match="finalized"):
        runtime.tool(name="late")


def test_decorator_captured_before_finalization_cannot_register_afterward() -> None:
    runtime = _runtime(_descriptor())
    captured_decorator = runtime.tool(name="late")

    @runtime.tool()
    async def echo() -> None:
        return None

    runtime.finalize()

    with pytest.raises(RuntimeError, match="finalized"):

        @captured_decorator
        async def late() -> None:
            return None
