"""Strict registration and raw dispatch tests for Chatbook's MCP gateway adapter."""

from __future__ import annotations

import asyncio
import copy
from functools import partial
import json
import threading
from typing import Any

import pytest
from loguru import logger

gateway = pytest.importorskip(
    "mcp_unified.gateway", reason="mcp-unified extra not installed"
)
GatewayRequestContext = gateway.GatewayRequestContext
GatewayToolExecutionError = gateway.GatewayToolExecutionError
GatewayProtocolConnection = gateway.GatewayProtocolConnection
GatewayLimits = gateway.GatewayLimits
PROTOCOL_PROFILES = gateway.PROTOCOL_PROFILES

from tldw_chatbook.MCP.gateway_runtime import ChatbookGatewayRuntime  # noqa: E402
from tldw_chatbook.Agents.agent_models import ToolResult  # noqa: E402
from tldw_chatbook.Agents.local_tool_provider import (  # noqa: E402
    LOCAL_DENY_REFUSAL,
    LOCAL_GATE_ERROR_REFUSAL,
    LOCAL_KILL_SWITCH_REFUSAL,
    LOCAL_TIMEOUT_REFUSAL,
)
from tldw_chatbook.DB.Subscriptions_DB import SubscriptionsDB  # noqa: E402
import tldw_chatbook.MCP.local_server_tools as local_server_tools  # noqa: E402
from tldw_chatbook.MCP.local_server_tools import (  # noqa: E402
    EXTERNAL_NO_CALLBACK_REFUSAL,
    LocalToolRegistration,
    _local_agent_tool_registrations,
    build_server_local_provider,
)
from tldw_chatbook.MCP.permission_store import (  # noqa: E402
    MCPPermissionStore,
    definition_hash,
)
from tldw_chatbook.runtime_policy.types import RuntimeSourceState  # noqa: E402
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
]
TASK_TOOL_NAMES = {"todo_create", "todo_update", "todo_get", "todo_list"}


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


def _runtime_with_builtins() -> ChatbookGatewayRuntime:
    runtime = _runtime(*_describe_local_tools())
    _register_real_builtins(runtime)
    return runtime


class _GatewayCoreToolHarness:
    """Supply unused core methods around the tool-only Chatbook runtime."""

    def __init__(self, runtime: ChatbookGatewayRuntime) -> None:
        self.name = runtime.name
        self.version = runtime.version
        self._runtime = runtime

    async def list_tools(self, context):
        return await self._runtime.list_tools(context)

    async def call_tool(self, name, arguments, context):
        return await self._runtime.call_tool(name, arguments, context)

    async def list_resources(self, _context):
        return []

    async def read_resource(self, _uri, _context):
        raise AssertionError("resources/read is not part of this tools/list test")

    async def list_prompts(self, _context):
        return []

    async def get_prompt(self, _name, _arguments, _context):
        raise AssertionError("prompts/get is not part of this tools/list test")


def _local_registration(
    name: str = "local_echo",
    *,
    parameters: object | None = None,
    handler: object | None = None,
    description: object = "Local echo tool",
) -> LocalToolRegistration:
    schema = (
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }
        if parameters is None
        else parameters
    )
    local_handler = (
        (lambda arguments: ToolResult(ok=True, content=arguments["value"]))
        if handler is None
        else handler
    )
    return LocalToolRegistration(
        name=name,
        description=description,  # type: ignore[arg-type]
        parameters=schema,  # type: ignore[arg-type]
        handler=local_handler,  # type: ignore[arg-type]
    )


def _schema_with_container_depth(depth: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "object", "properties": {}}
    current = schema
    for _ in range(1, depth):
        child: dict[str, Any] = {}
        current["default"] = child
        current = child
    return schema


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


@pytest.mark.parametrize(
    "explicit_name",
    [pytest.param("", id="empty-string"), pytest.param(0, id="zero")],
)
def test_runtime_rejects_falsey_explicit_handler_name(explicit_name: object) -> None:
    runtime = _runtime(_descriptor())

    with pytest.raises(ValueError) as exc_info:

        @runtime.tool(name=explicit_name)  # type: ignore[arg-type]
        async def echo() -> None:
            return None

    assert str(exc_info.value) == "tool handler name is invalid"
    assert len(str(exc_info.value)) <= 512


def test_runtime_derives_handler_name_only_when_name_is_none() -> None:
    runtime = _runtime(_descriptor("derived"))

    @runtime.tool(name=None)
    async def derived() -> None:
        return None

    runtime.finalize()

    assert runtime._tool_handlers["derived"] is derived


def test_runtime_rejects_implicit_name_for_coroutine_callable_without_name() -> None:
    runtime = _runtime(_descriptor())

    async def handler() -> None:
        return None

    nameless_handler = partial(handler)
    assert not hasattr(nameless_handler, "__name__")

    with pytest.raises(ValueError) as exc_info:
        runtime.tool()(nameless_handler)

    assert str(exc_info.value) == "tool handler name is invalid"
    assert len(str(exc_info.value)) <= 512


@pytest.mark.asyncio
async def test_coroutine_callable_without_name_registers_with_explicit_name() -> None:
    runtime = _runtime(_descriptor("join"))

    async def handler(left: str, right: str) -> str:
        return left + right

    nameless_handler = partial(handler, "chat")
    assert not hasattr(nameless_handler, "__name__")

    registered = runtime.tool(name="join")(nameless_handler)
    runtime.finalize()

    assert registered is nameless_handler
    assert await runtime.call_tool("join", {"right": "book"}, _context()) == "chatbook"


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
async def test_all_nine_builtin_handlers_register_with_exact_names() -> None:
    runtime = _runtime(*_describe_local_tools())
    _register_real_builtins(runtime)
    runtime.finalize()

    descriptors = await runtime.list_tools(_context())

    assert [descriptor["name"] for descriptor in descriptors] == BUILTIN_TOOL_NAMES
    assert "ingest_media" not in runtime._tool_handlers
    assert list(runtime._tool_handlers) == BUILTIN_TOOL_NAMES
    assert [handler.__name__ for handler in runtime._tool_handlers.values()] == (
        BUILTIN_TOOL_NAMES
    )


def test_all_nine_builtin_schemas_reject_additional_properties() -> None:
    descriptors = _describe_local_tools()

    assert [descriptor["name"] for descriptor in descriptors] == BUILTIN_TOOL_NAMES
    assert "ingest_media" not in {descriptor["name"] for descriptor in descriptors}
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


LOCAL_FAILURES = [
    pytest.param(
        EXTERNAL_NO_CALLBACK_REFUSAL,
        "operator_approval_required",
        "Operator approval is required for this local tool.",
        id="external-ask",
    ),
    pytest.param(
        LOCAL_TIMEOUT_REFUSAL,
        "operator_approval_required",
        "Operator approval is required for this local tool.",
        id="approval-timeout",
    ),
    pytest.param(
        LOCAL_DENY_REFUSAL,
        "tool_permission_denied",
        "This local tool is disabled by operator policy.",
        id="operator-deny",
    ),
    pytest.param(
        LOCAL_KILL_SWITCH_REFUSAL,
        "local_tools_disabled",
        "Local tools are disabled.",
        id="kill-switch",
    ),
    pytest.param(
        LOCAL_GATE_ERROR_REFUSAL,
        "permission_state_unavailable",
        "Local tool permission state is unavailable.",
        id="permission-store-error",
    ),
    pytest.param(
        "SENTINEL /private/path API_KEY=secret",
        "local_tool_failed",
        "Local tool execution failed.",
        id="provider-failure",
    ),
]


@pytest.mark.parametrize("provider_error,reason_code,public_message", LOCAL_FAILURES)
@pytest.mark.asyncio
async def test_local_failure_mapping_is_exact_and_payload_free(
    provider_error: str,
    reason_code: str,
    public_message: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    runtime = _runtime_with_builtins()
    runtime.register_local_tools(
        [
            _local_registration(
                handler=lambda _arguments: ToolResult(ok=False, error=provider_error)
            )
        ]
    )
    runtime.finalize()
    log_records: list[str] = []
    sink_id = logger.add(lambda message: log_records.append(str(message)))

    try:
        with pytest.raises(GatewayToolExecutionError) as exc_info:
            await runtime.call_tool("local_echo", {"value": "raw argument"}, _context())
    finally:
        logger.remove(sink_id)

    error = exc_info.value
    assert error.kind == "tool"
    assert error.reason_code == reason_code
    assert error.public_message == public_message
    public_exception = " ".join((str(error), repr(error), repr(vars(error))))
    captured = capsys.readouterr()
    private_values = {
        provider_error,
        "SENTINEL",
        "/private/path",
        "API_KEY=secret",
        "raw argument",
    }
    for private_value in private_values:
        assert private_value not in public_exception
        assert private_value not in captured.out
        assert private_value not in captured.err
        assert all(private_value not in record for record in log_records)


@pytest.mark.asyncio
async def test_raised_local_provider_exception_maps_to_fixed_payload_free_error(
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "SENTINEL /private/path API_KEY=secret"

    def raise_private_exception(_arguments: dict[str, Any]) -> ToolResult:
        raise RuntimeError(sentinel)

    runtime = _runtime_with_builtins()
    runtime.register_local_tools([_local_registration(handler=raise_private_exception)])
    runtime.finalize()
    log_records: list[str] = []
    sink_id = logger.add(lambda message: log_records.append(str(message)))

    try:
        with pytest.raises(GatewayToolExecutionError) as exc_info:
            await runtime.call_tool("local_echo", {"value": sentinel}, _context())
    finally:
        logger.remove(sink_id)

    error = exc_info.value
    assert error.reason_code == "local_tool_failed"
    assert error.public_message == "Local tool execution failed."
    assert error.__context__ is None
    assert error.__cause__ is None
    captured = capsys.readouterr()
    assert sentinel not in " ".join((str(error), repr(error), repr(vars(error))))
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert all(sentinel not in record for record in log_records)


@pytest.mark.parametrize("protocol_version", PROTOCOL_PROFILES)
@pytest.mark.asyncio
async def test_real_provider_schemas_compile_for_every_public_profile(
    tmp_path, protocol_version: str
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    provider = build_server_local_provider(
        workspace, MCPPermissionStore(tmp_path / "mcp_permissions.json")
    )
    registrations = _local_agent_tool_registrations(provider)
    expected_schemas = {
        registration.name: copy.deepcopy(registration.parameters)
        for registration in registrations
    }
    assert len(expected_schemas) == 17
    assert all("$schema" not in schema for schema in expected_schemas.values())
    runtime = _runtime_with_builtins()
    runtime.register_local_tools(registrations)
    runtime.finalize()
    responses: list[Any] = []

    async def write_response(response: Any) -> None:
        responses.append(copy.deepcopy(response))

    connection = GatewayProtocolConnection(
        _GatewayCoreToolHarness(runtime), write_response
    )
    profile = PROTOCOL_PROFILES[protocol_version]
    try:
        if profile.requires_initialize:
            await connection.receive(
                {
                    "jsonrpc": "2.0",
                    "id": "initialize",
                    "method": "initialize",
                    "params": {
                        "protocolVersion": protocol_version,
                        "capabilities": {},
                        "clientInfo": {"name": "schema-test", "version": "1"},
                    },
                }
            )
            await connection.wait_for_idle()
            assert responses[-1]["result"]["protocolVersion"] == protocol_version
            await connection.receive(
                {
                    "jsonrpc": "2.0",
                    "method": "notifications/initialized",
                    "params": {},
                }
            )
            list_params = {}
        else:
            list_params = {
                "_meta": {
                    "io.modelcontextprotocol/protocolVersion": protocol_version,
                    "io.modelcontextprotocol/clientCapabilities": {},
                    "io.modelcontextprotocol/clientInfo": {
                        "name": "schema-test",
                        "version": "1",
                    },
                }
            }

        await connection.receive(
            {
                "jsonrpc": "2.0",
                "id": "tools-list",
                "method": "tools/list",
                "params": list_params,
            }
        )
        await connection.wait_for_idle()
    finally:
        await connection.shutdown()

    tools_response = next(
        response for response in responses if response.get("id") == "tools-list"
    )
    assert "error" not in tools_response
    tools = tools_response["result"]["tools"]
    published_names = {descriptor["name"] for descriptor in tools}
    assert len(tools) == len(BUILTIN_TOOL_NAMES) + len(expected_schemas)
    published_locals = {
        descriptor["name"]: descriptor["inputSchema"]
        for descriptor in tools
        if descriptor["name"] in expected_schemas
    }
    assert published_locals == expected_schemas
    assert "todo_write" not in published_names
    assert TASK_TOOL_NAMES.isdisjoint(published_names)


@pytest.mark.asyncio
async def test_local_tool_schema_copies_are_detached() -> None:
    schema = {
        "type": "object",
        "properties": {"value": {"type": "string"}},
        "required": ["value"],
    }
    expected = copy.deepcopy(schema)
    runtime = _runtime_with_builtins()
    runtime.register_local_tools([_local_registration(parameters=schema)])
    schema["properties"]["value"]["type"] = "integer"
    runtime.finalize()

    first_listing = await runtime.list_tools(_context())
    first_local = next(
        descriptor for descriptor in first_listing if descriptor["name"] == "local_echo"
    )
    assert first_local["inputSchema"] == expected
    first_local["description"] = "mutated"
    first_local["inputSchema"]["properties"]["value"]["type"] = "boolean"

    second_listing = await runtime.list_tools(_context())
    second_local = next(
        descriptor
        for descriptor in second_listing
        if descriptor["name"] == "local_echo"
    )
    assert second_local["description"] == "Local echo tool"
    assert second_local["inputSchema"] == expected


@pytest.mark.asyncio
async def test_successful_local_tool_result_content_is_returned_raw() -> None:
    content = "provider content"
    runtime = _runtime_with_builtins()
    runtime.register_local_tools(
        [
            _local_registration(
                handler=lambda _arguments: ToolResult(ok=True, content=content)
            )
        ]
    )
    runtime.finalize()

    result = await runtime.call_tool("local_echo", {"value": "ignored"}, _context())

    assert result is content


@pytest.mark.parametrize(
    "registrations",
    [
        pytest.param(
            [_local_registration("duplicate"), _local_registration("duplicate")],
            id="duplicate-local-name",
        ),
        pytest.param(
            [_local_registration(BUILTIN_TOOL_NAMES[0])],
            id="built-in-collision",
        ),
        pytest.param(
            [_local_registration(parameters={"type": "array"})],
            id="non-object-schema",
        ),
        pytest.param(
            [
                _local_registration(
                    parameters={
                        "type": "object",
                        "properties": [],
                        "required": [],
                    }
                )
            ],
            id="malformed-object-schema",
        ),
        pytest.param(
            [
                _local_registration(
                    parameters={
                        "type": "object",
                        "properties": {},
                        "default": object(),
                    }
                )
            ],
            id="non-json-schema-value",
        ),
        pytest.param(
            [
                _local_registration(
                    parameters={
                        "$schema": "http://json-schema.org/draft-07/schema#",
                        "type": "object",
                        "properties": {},
                    }
                )
            ],
            id="explicit-draft-7",
        ),
        pytest.param(
            [
                _local_registration(
                    parameters={
                        "$schema": "https://json-schema.org/draft/2020-12/schema",
                        "type": "object",
                        "properties": {},
                    }
                )
            ],
            id="explicit-draft-2020-12",
        ),
        pytest.param(
            [
                _local_registration(
                    parameters={
                        "type": "object",
                        "properties": {
                            "values": {
                                "type": "array",
                                "items": [{"type": "string"}],
                            }
                        },
                    }
                )
            ],
            id="draft-2020-12-incompatible-schema",
        ),
        pytest.param(
            [_local_registration(handler="not callable")],
            id="non-callable-handler",
        ),
        pytest.param(
            [
                _local_registration("valid_first"),
                _local_registration("invalid_second", parameters={"type": "array"}),
            ],
            id="mid-list-invalid",
        ),
        pytest.param(
            [
                _local_registration("valid_first"),
                _local_registration(
                    "invalid_second",
                    parameters={
                        "type": "object",
                        "properties": [],
                        "required": [],
                    },
                ),
            ],
            id="mid-list-malformed-object-schema",
        ),
        pytest.param(
            [
                _local_registration("valid_first"),
                _local_registration(
                    "draft-7-incompatible",
                    parameters={
                        "type": "object",
                        "properties": {},
                        "additionalItems": 1,
                    },
                ),
            ],
            id="mid-list-draft-7-incompatible-schema",
        ),
        pytest.param(
            [_local_registration(description="")],
            id="invalid-description",
        ),
    ],
)
@pytest.mark.asyncio
async def test_invalid_local_registration_is_atomic(
    registrations: list[LocalToolRegistration],
) -> None:
    await _assert_invalid_local_registration_is_atomic(registrations)


async def _assert_invalid_local_registration_is_atomic(
    registrations: list[LocalToolRegistration],
) -> ValueError:
    runtime = _runtime_with_builtins()

    with pytest.raises(ValueError) as exc_info:
        runtime.register_local_tools(registrations)

    assert str(exc_info.value) in {
        "local tool description must be a bounded string",
        "local tool handler must be callable",
        "local tool name collides with another tool",
        "local tool parameters must have type object",
        "local tool parameters must be valid JSON Schema",
        "local tool parameters must not declare a schema dialect",
    }
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    assert list(runtime._tool_descriptors) == BUILTIN_TOOL_NAMES
    assert list(runtime._tool_handlers) == BUILTIN_TOOL_NAMES
    runtime.finalize()
    assert [
        descriptor["name"] for descriptor in await runtime.list_tools(_context())
    ] == BUILTIN_TOOL_NAMES
    return exc_info.value


@pytest.mark.asyncio
async def test_unpaired_surrogate_schema_is_rejected_atomically() -> None:
    raw_value = "\ud800"
    error = await _assert_invalid_local_registration_is_atomic(
        [
            _local_registration(
                parameters={
                    "type": "object",
                    "properties": {},
                    "description": raw_value,
                }
            )
        ]
    )

    assert raw_value not in str(error)


@pytest.mark.asyncio
async def test_schema_beyond_public_depth_limit_is_rejected_atomically() -> None:
    limits = GatewayLimits()
    assert limits.max_schema_depth < 40

    await _assert_invalid_local_registration_is_atomic(
        [_local_registration(parameters=_schema_with_container_depth(40))]
    )


@pytest.mark.asyncio
async def test_schema_above_public_utf8_size_limit_is_rejected_atomically() -> None:
    limits = GatewayLimits()
    raw_value = "é" * (limits.max_schema_bytes // 2)
    schema = {
        "type": "object",
        "properties": {},
        "description": raw_value,
    }
    serialized = json.dumps(
        schema,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    assert len(serialized) < limits.max_schema_bytes
    assert len(serialized.encode("utf-8")) > limits.max_schema_bytes

    error = await _assert_invalid_local_registration_is_atomic(
        [_local_registration(parameters=schema)]
    )
    assert raw_value not in str(error)


@pytest.mark.parametrize(
    "schema",
    [
        pytest.param(
            _schema_with_container_depth(GatewayLimits().max_schema_depth),
            id="maximum-depth",
        ),
        pytest.param(
            {
                "type": "object",
                "properties": {},
                "description": "Normal Unicode: café 🦄",
            },
            id="normal-unicode",
        ),
    ],
)
@pytest.mark.asyncio
async def test_schema_structure_boundary_controls_publish_unchanged(
    schema: dict[str, Any],
) -> None:
    expected = copy.deepcopy(schema)
    runtime = _runtime_with_builtins()
    runtime.register_local_tools([_local_registration(parameters=schema)])
    runtime.finalize()

    published = await runtime.list_tools(_context())
    local = next(item for item in published if item["name"] == "local_echo")
    assert local["inputSchema"] == expected


@pytest.mark.asyncio
async def test_blocking_local_handler_runs_off_event_loop() -> None:
    entered = threading.Event()
    release = threading.Event()

    def blocking_handler(_arguments: dict[str, Any]) -> ToolResult:
        entered.set()
        assert release.wait(timeout=2)
        return ToolResult(ok=True, content="done")

    runtime = _runtime_with_builtins()
    runtime.register_local_tools([_local_registration(handler=blocking_handler)])
    runtime.finalize()
    heartbeat = 0

    async def beat() -> None:
        nonlocal heartbeat
        while not release.is_set():
            heartbeat += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(beat())
    call_task = asyncio.create_task(
        runtime.call_tool("local_echo", {"value": "ignored"}, _context())
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.sleep(0.02)
        assert heartbeat > 1
    finally:
        release.set()

    assert await call_task == "done"
    await heartbeat_task


@pytest.mark.asyncio
async def test_real_watchlists_provider_preserves_structured_domain_outcomes(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    db_path = tmp_path / "subscriptions.db"
    mutable = SubscriptionsDB(db_path)
    mutable.close()
    source = {"value": "local"}

    monkeypatch.setattr(
        local_server_tools, "get_subscriptions_db_path", lambda: db_path
    )
    _pin_runtime_source(monkeypatch, lambda: source["value"])
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    provider = build_server_local_provider(workspace, store)
    _grant_local_tool(store, provider, "watchlists_search_items")
    _grant_local_tool(store, provider, "watchlists_get_item")
    runtime = _runtime_with_builtins()
    runtime.register_local_tools(_local_agent_tool_registrations(provider))
    runtime.finalize()

    calls = [
        ("watchlists_search_items", {"limit": True}, "invalid_argument"),
        (
            "watchlists_get_item",
            {"item_id": "local:watchlist_item:999"},
            "not_found",
        ),
    ]
    for name, arguments, status in calls:
        expected = provider.invoke(f"local:{name}", arguments)
        actual = await runtime.call_tool(name, arguments, _context())
        assert actual == expected.content
        assert json.loads(actual)["status"] == status

    source["value"] = "server"
    arguments = {}
    expected = provider.invoke("local:watchlists_search_items", arguments)
    actual = await runtime.call_tool("watchlists_search_items", arguments, _context())
    assert actual == expected.content
    assert json.loads(actual)["status"] == "unsupported"

    source["value"] = "local"
    missing_store = MCPPermissionStore(tmp_path / "missing-permissions.json")
    missing_provider = build_server_local_provider(workspace, missing_store)
    _grant_local_tool(missing_store, missing_provider, "watchlists_search_items")
    monkeypatch.setattr(
        local_server_tools,
        "get_subscriptions_db_path",
        lambda: tmp_path / "does-not-exist.db",
    )
    missing_runtime = _runtime_with_builtins()
    missing_runtime.register_local_tools(
        _local_agent_tool_registrations(missing_provider)
    )
    missing_runtime.finalize()
    expected = missing_provider.invoke("local:watchlists_search_items", {})
    actual = await missing_runtime.call_tool("watchlists_search_items", {}, _context())
    assert actual == expected.content
    assert json.loads(actual)["status"] == "feature_unavailable"


@pytest.mark.asyncio
async def test_real_watchlists_gateway_permission_failures_precede_storage(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    path_calls = 0

    def resolve_path():
        nonlocal path_calls
        path_calls += 1
        raise AssertionError("permission failures must precede storage")

    monkeypatch.setattr(local_server_tools, "get_subscriptions_db_path", resolve_path)
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    provider = build_server_local_provider(workspace, store)
    runtime = _runtime_with_builtins()
    runtime.register_local_tools(_local_agent_tool_registrations(provider))
    runtime.finalize()

    with pytest.raises(GatewayToolExecutionError) as ask_error:
        await runtime.call_tool("watchlists_search_items", {}, _context())
    assert ask_error.value.reason_code == "operator_approval_required"
    assert path_calls == 0

    hub = provider.hub_tool_for("watchlists_search_items")
    store.set_tool_state(hub.server_key, hub.name, "deny")
    with pytest.raises(GatewayToolExecutionError) as deny_error:
        await runtime.call_tool("watchlists_search_items", {}, _context())
    assert deny_error.value.reason_code == "tool_permission_denied"
    assert path_calls == 0


@pytest.mark.asyncio
async def test_real_watchlists_provider_scrubs_unexpected_failures(
    monkeypatch, tmp_path, capsys, caplog
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sentinel = "SENTINEL /private/db.sqlite API_KEY=secret"

    def fail_database(*_args, **_kwargs):
        raise RuntimeError(sentinel)

    monkeypatch.setattr(
        local_server_tools, "get_subscriptions_db_path", lambda: tmp_path / "db.sqlite"
    )
    _pin_runtime_source(monkeypatch, "local")
    monkeypatch.setattr(local_server_tools, "SubscriptionsDB", fail_database)
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    provider = build_server_local_provider(workspace, store)
    _grant_local_tool(store, provider, "watchlists_search_items")
    runtime = _runtime_with_builtins()
    runtime.register_local_tools(_local_agent_tool_registrations(provider))
    runtime.finalize()
    records: list[str] = []
    sink_id = logger.add(lambda message: records.append(str(message)))

    try:
        with pytest.raises(GatewayToolExecutionError) as exc_info:
            await runtime.call_tool("watchlists_search_items", {}, _context())
    finally:
        logger.remove(sink_id)

    assert exc_info.value.reason_code == "local_tool_failed"
    assert exc_info.value.public_message == "Local tool execution failed."
    captured = capsys.readouterr()
    assert sentinel not in str(exc_info.value)
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert all(sentinel not in record for record in records)
    # TASK-19569: the stdlib-`logging` channel was NOT covered here, and
    # `WatchlistsToolService._raise_unexpected` -- the scrubber on this very
    # path -- logs through `logging.getLogger(__name__)`, not loguru. Adding
    # `detail=%s` to that call leaked the sentinel into the captured log and
    # this test still passed; the loguru sink and capsys never see stdlib
    # records. The sibling guard in `Tests/MCP/test_local_server_tools.py`
    # (`..._blocks_replacement_until_failed_close_succeeds`) already asserts
    # against `caplog.text`; this one now does too.
    assert sentinel not in caplog.text


@pytest.mark.asyncio
async def test_real_watchlists_database_resolution_runs_off_event_loop(
    monkeypatch, tmp_path
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    entered = threading.Event()
    release = threading.Event()

    class BlockingDatabase:
        def __init__(self):
            self.readiness_calls = 0

        def assert_agent_read_ready(self):
            self.readiness_calls += 1
            if self.readiness_calls == 1:
                entered.set()
                assert release.wait(timeout=2)

        def search_items_for_agent(self, **_kwargs):
            return {"items": [], "has_more": False, "snapshot_max_item_id": 0}

        def get_source_collection_memberships(self, _source_ids):
            return {}

        def close(self):
            return None

    database = BlockingDatabase()
    monkeypatch.setattr(
        local_server_tools, "get_subscriptions_db_path", lambda: tmp_path / "db.sqlite"
    )
    _pin_runtime_source(monkeypatch, "local")
    monkeypatch.setattr(
        local_server_tools,
        "SubscriptionsDB",
        lambda *_args, **_kwargs: database,
    )
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    provider = build_server_local_provider(workspace, store)
    _grant_local_tool(store, provider, "watchlists_search_items")
    runtime = _runtime_with_builtins()
    runtime.register_local_tools(_local_agent_tool_registrations(provider))
    runtime.finalize()
    heartbeat = 0

    async def beat() -> None:
        nonlocal heartbeat
        while not release.is_set():
            heartbeat += 1
            await asyncio.sleep(0)

    heartbeat_task = asyncio.create_task(beat())
    call_task = asyncio.create_task(
        runtime.call_tool("watchlists_search_items", {}, _context())
    )
    try:
        assert await asyncio.to_thread(entered.wait, 1)
        await asyncio.sleep(0.02)
        assert heartbeat > 1
    finally:
        release.set()

    result = await call_task
    await heartbeat_task
    assert json.loads(result)["status"] == "ok"


def _pin_runtime_source(monkeypatch, source) -> None:
    """Pin the runtime source the composed watchlists service will read.

    The seam is ``local_server_tools.load_default_runtime_source_state`` --
    the owner-module loader ``build_server_local_provider`` injects as
    ``runtime_source_loader=`` (TASK-18609). These tests kept patching the
    ``RuntimeSourceStateStore`` name it replaced, which no longer exists on
    the module, so all three errored at the monkeypatch line (TASK-19569).

    ``source`` may be a literal ``"local"``/``"server"`` or a zero-arg
    callable, so a test can flip the source between gateway calls. The
    loader returns a real ``RuntimeSourceState`` -- production's shape.

    Deliberately NOT ``raising=False``: a renamed seam must fail loudly at
    the patch line rather than silently install a never-read attribute.
    """
    resolve = source if callable(source) else (lambda: source)
    monkeypatch.setattr(
        local_server_tools,
        "load_default_runtime_source_state",
        lambda: RuntimeSourceState(active_source=resolve()),
    )


def _grant_local_tool(
    store: MCPPermissionStore, provider: Any, name: str = "fs_read"
) -> None:
    hub = provider.hub_tool_for(name)
    store.set_tool_state(
        hub.server_key,
        hub.name,
        "allow",
        definition_hash=definition_hash(hub.description, hub.input_schema),
    )


async def _assert_local_refusal(
    runtime: ChatbookGatewayRuntime, reason_code: str
) -> None:
    with pytest.raises(GatewayToolExecutionError) as exc_info:
        await runtime.call_tool("fs_read", {"path": "hello.txt"}, _context())
    assert exc_info.value.reason_code == reason_code


@pytest.mark.asyncio
async def test_running_adapter_reloads_permissions_and_kill_switch_each_call(
    tmp_path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "hello.txt").write_text("hello world\n", encoding="utf-8")
    store = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    provider = build_server_local_provider(workspace, store)
    runtime = _runtime_with_builtins()
    runtime.register_local_tools(_local_agent_tool_registrations(provider))
    runtime.finalize()

    await _assert_local_refusal(runtime, "operator_approval_required")
    _grant_local_tool(store, provider)
    assert "hello world" in await runtime.call_tool(
        "fs_read", {"path": "hello.txt"}, _context()
    )

    hub = provider.hub_tool_for("fs_read")
    store.set_tool_state(hub.server_key, hub.name, "deny")
    await _assert_local_refusal(runtime, "tool_permission_denied")
    _grant_local_tool(store, provider)
    assert "hello world" in await runtime.call_tool(
        "fs_read", {"path": "hello.txt"}, _context()
    )

    store.set_tool_state(hub.server_key, hub.name, None)
    await _assert_local_refusal(runtime, "operator_approval_required")
    _grant_local_tool(store, provider)

    store.set_kill_switch(False)
    assert "hello world" in await runtime.call_tool(
        "fs_read", {"path": "hello.txt"}, _context()
    )
    store.set_kill_switch(True)
    await _assert_local_refusal(runtime, "local_tools_disabled")
    store.set_kill_switch(False)
    assert "hello world" in await runtime.call_tool(
        "fs_read", {"path": "hello.txt"}, _context()
    )
