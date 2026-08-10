"""Bounded catalog pagination and exact resource metadata client contracts."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from itertools import count
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from tldw_chatbook.MCP import client as client_module


MAX_OUTPUT_LINE_BYTES = 1_048_576


class _Reader:
    def __init__(self, *lines: bytes) -> None:
        self.lines = list(lines)

    async def readline(self) -> bytes:
        return self.lines.pop(0) if self.lines else b""


class _Stdin:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None


class _Process:
    def __init__(self, stdout: object | None = None) -> None:
        self.stdout = stdout
        self.stderr = None
        self.stdin = _Stdin()
        self.returncode: int | None = None
        self.terminate_calls = 0
        self.kill_calls = 0
        self.wait_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1
        self.returncode = 0

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        return self.returncode or 0


class _SlowReapProcess(_Process):
    def __init__(self, stdout: object | None = None) -> None:
        super().__init__(stdout)
        self.wait_started = asyncio.Event()
        self.allow_reap = asyncio.Event()

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.returncode = -9
        self.allow_reap.set()

    async def wait(self) -> int:
        self.wait_calls += 1
        self.wait_started.set()
        await self.allow_reap.wait()
        if self.returncode is None:
            self.returncode = 0
        return self.returncode


def _bare_connection(
    process: object | None = None,
) -> client_module._StdioJSONRPCConnection:
    connection = client_module._StdioJSONRPCConnection.__new__(
        client_module._StdioJSONRPCConnection
    )
    connection.process = process or _Process()
    connection._request_ids = count(1)
    connection._pending_requests = {}
    connection._write_lock = asyncio.Lock()
    connection._reader_unavailable = False
    connection._cleanup_complete = False
    connection._read_task = None
    connection._stderr_task = None
    connection._on_transport_failure = None
    connection._transport_cleanup_task = None
    connection.request_timeout_seconds = 10
    return connection


CATALOG_RESPONSES = [
    pytest.param("list_tools", "tools", "tools/list", "name", id="tools"),
    pytest.param(
        "list_resources", "resources", "resources/list", "uri", id="resources"
    ),
    pytest.param("list_prompts", "prompts", "prompts/list", "name", id="prompts"),
]


def _item(item_key: str, value: str) -> dict[str, Any]:
    if item_key == "resources":
        return {"uri": f"resource:{value}", "name": value}
    if item_key == "tools":
        return {"name": value, "inputSchema": {"type": "object"}}
    return {"name": value}


def _scripted_connection(
    responder: Callable[[int, str, dict[str, Any]], dict[str, Any]],
) -> tuple[client_module._StdioJSONRPCConnection, list[tuple[str, dict[str, Any]]]]:
    connection = client_module._StdioJSONRPCConnection.__new__(
        client_module._StdioJSONRPCConnection
    )
    requests: list[tuple[str, dict[str, Any]]] = []

    async def request(
        method: str,
        params: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        del timeout_seconds
        copied_params = dict(params)
        requests.append((method, copied_params))
        return responder(len(requests), method, copied_params)

    connection.request = request  # type: ignore[method-assign]
    return connection, requests


async def _catalog_values(
    connection: client_module._StdioJSONRPCConnection,
    list_method: str,
    item_key: str,
    value_field: str,
) -> list[str]:
    response = await getattr(connection, list_method)()
    return [getattr(item, value_field) for item in getattr(response, item_key)]


def _assert_client_error(error: BaseException, expected_message: str) -> None:
    assert type(error).__name__ == "MCPClientError"
    assert str(error) == expected_message


@pytest.mark.parametrize(
    "protocol_version",
    [
        pytest.param(None, id="missing"),
        pytest.param("", id="empty"),
        pytest.param(7, id="non-string"),
        pytest.param("2025-11-25", id="mismatch"),
        pytest.param("private-version" * 100, id="oversized"),
    ],
)
@pytest.mark.asyncio
async def test_initialize_rejects_unexpected_protocol_version_without_payload_leakage(
    protocol_version: object,
) -> None:
    connection = client_module._StdioJSONRPCConnection.__new__(
        client_module._StdioJSONRPCConnection
    )
    connection.client_name = "test-client"

    async def request(_method: str, _params: dict[str, Any]) -> dict[str, Any]:
        return {"protocolVersion": protocol_version}

    async def notify(_method: str) -> None:
        pytest.fail("initialized notification must not follow invalid negotiation")

    connection.request = request  # type: ignore[method-assign]
    connection.notify = notify  # type: ignore[method-assign]

    with pytest.raises(Exception) as exc_info:
        await connection.initialize()

    _assert_client_error(exc_info.value, "Unexpected MCP protocol version")
    assert "private-version" not in str(exc_info.value)


@pytest.mark.parametrize("phase", ["initialize", "discovery"])
@pytest.mark.asyncio
async def test_connect_cancellation_closes_child_and_removes_partial_session(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    started = asyncio.Event()
    process = _Process()

    class Session:
        protocol_version = "2025-03-26"
        server_info: dict[str, Any] = {}
        server_capabilities: dict[str, Any] = {}

        def __init__(self, created_process: object, *, client_name: str) -> None:
            assert created_process is process
            assert client_name == "cancel-client"
            self.process = created_process
            self.closed = False

        async def initialize(self) -> None:
            if phase == "initialize":
                started.set()
                await asyncio.Future()

        async def close(self) -> None:
            self.closed = True
            process.stdin.close()
            process.terminate()
            await process.wait()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    async def discover(_server_id: str) -> None:
        if phase == "discovery":
            started.set()
            await asyncio.Future()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)
    client = client_module.MCPClient(name="cancel-client")
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)

    task = asyncio.create_task(client.connect_to_server("server", "python"))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.returncode is not None
    assert process.stdin.closed
    assert client.sessions == {}
    assert client.servers == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("terminate_raises", [False, True])
async def test_connect_cancellation_forces_hung_cleanup_and_removes_state(
    terminate_raises: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_started = asyncio.Event()

    class SlowProcess(_Process):
        def terminate(self) -> None:
            self.terminate_calls += 1
            if terminate_raises:
                raise RuntimeError("terminate failed")

        async def wait(self) -> int:
            self.wait_calls += 1
            if not self.kill_calls:
                await asyncio.Future()
            return self.returncode or 0

    process = SlowProcess()

    class Session:
        protocol_version = "2025-03-26"
        server_info: dict[str, Any] = {}
        server_capabilities: dict[str, Any] = {}

        def __init__(self, created_process: object, *, client_name: str) -> None:
            assert created_process is process
            self.process = created_process

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            await asyncio.Future()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    async def discover(_server_id: str) -> None:
        discovery_started.set()
        await asyncio.Future()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)
    client = client_module.MCPClient(name="hung-cleanup-client")
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)

    task = asyncio.create_task(client.connect_to_server("server", "python"))
    await asyncio.wait_for(discovery_started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert process.returncode is not None
    assert process.kill_calls == 1
    assert process.wait_calls == 2
    assert client.sessions == {}
    assert client.servers == {}


@pytest.mark.asyncio
async def test_connect_deeply_detaches_server_metadata_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process()
    connection = SimpleNamespace(
        process=process,
        protocol_version="2025-03-26",
        server_info={"nested": {"value": "original"}},
        server_capabilities={"tools": {"listChanged": True}},
    )

    async def no_op(*_args: Any) -> None:
        return None

    connection.initialize = no_op
    connection.close = no_op

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(
        client_module,
        "_StdioJSONRPCConnection",
        lambda *_args, **_kwargs: connection,
    )
    client = client_module.MCPClient(name="detached-cache-client")
    monkeypatch.setattr(client, "_discover_server_capabilities", no_op)

    assert await client.connect_to_server("server", "python") is True
    connection.server_info["nested"]["value"] = "session-mutation"
    connection.server_capabilities["tools"]["listChanged"] = False

    assert client.servers["server"]["server_info"] == {"nested": {"value": "original"}}
    assert client.servers["server"]["server_capabilities"] == {
        "tools": {"listChanged": True}
    }


@pytest.mark.asyncio
async def test_request_cancellation_always_removes_pending_request() -> None:
    connection = _bare_connection()

    async def send(_payload: dict[str, Any]) -> None:
        return None

    connection._send_message = send  # type: ignore[method-assign]
    task = asyncio.create_task(connection.request("tools/list"))
    await asyncio.sleep(0)
    assert list(connection._pending_requests) == [1]

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert connection._pending_requests == {}


@pytest.mark.asyncio
async def test_bulk_failure_removes_done_cancelled_and_pending_requests() -> None:
    connection = _bare_connection()
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    done.set_result({})
    cancelled = loop.create_future()
    cancelled.cancel()
    pending = loop.create_future()
    connection._pending_requests = {1: done, 2: cancelled, 3: pending}
    failure = RuntimeError("bounded failure")

    connection._fail_pending_requests(failure)

    assert connection._pending_requests == {}
    assert pending.exception() is failure


@pytest.mark.asyncio
async def test_eof_marks_reader_unavailable_but_close_still_cleans_live_child() -> None:
    process = _Process(_Reader(b""))
    connection = client_module._StdioJSONRPCConnection(
        process, client_name="transport-client"
    )

    await connection._read_task
    assert connection._reader_unavailable is True
    assert connection._cleanup_complete is False

    await connection.close()
    await connection.close()

    assert process.stdin.closed
    assert process.terminate_calls == 1
    assert process.wait_calls == 1
    assert connection._cleanup_complete is True


@pytest.mark.asyncio
async def test_catalog_uses_one_monotonic_deadline_and_remaining_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 0.25, 1.0])
    monkeypatch.setattr(client_module, "_monotonic", lambda: next(clock), raising=False)
    monkeypatch.setattr(client_module, "CATALOG_TIMEOUT_SECONDS", 1.0, raising=False)
    connection = _bare_connection()
    timeouts: list[float | None] = []

    async def request(
        _method: str,
        _params: dict[str, Any],
        *,
        timeout_seconds: float | None = None,
    ) -> dict[str, Any]:
        timeouts.append(timeout_seconds)
        return {
            "tools": [{"name": "one", "inputSchema": {"type": "object"}}],
            "nextCursor": "more",
        }

    connection.request = request  # type: ignore[method-assign]

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "MCP catalog deadline exceeded")
    assert timeouts == [0.75]


@pytest.mark.parametrize("timeout", [0.0, -1.0])
@pytest.mark.asyncio
async def test_catalog_rejects_nonpositive_aggregate_deadline_before_request(
    timeout: float,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_monotonic", lambda: 1.0, raising=False)
    monkeypatch.setattr(
        client_module, "CATALOG_TIMEOUT_SECONDS", timeout, raising=False
    )
    connection = _bare_connection()

    async def request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        pytest.fail("expired catalog must not issue a request")

    connection.request = request  # type: ignore[method-assign]

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "MCP catalog deadline exceeded")


@pytest.mark.asyncio
async def test_connect_uses_one_deadline_and_cleans_near_timeout_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter([0.0, 0.0, 2.0])
    monkeypatch.setattr(client_module, "_monotonic", lambda: next(clock), raising=False)
    monkeypatch.setattr(client_module, "CONNECT_TIMEOUT_SECONDS", 1.0, raising=False)
    process = _Process()

    class Session:
        protocol_version = "2025-03-26"
        server_info: dict[str, Any] = {}
        server_capabilities: dict[str, Any] = {}

        def __init__(self, created_process: object, *, client_name: str) -> None:
            assert created_process is process
            self.process = created_process

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            process.stdin.close()
            process.terminate()
            await process.wait()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    async def discover(_server_id: str) -> None:
        return None

    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)
    client = client_module.MCPClient(name="deadline-client")
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)

    assert await client.connect_to_server("server", "python") is False
    assert process.returncode is not None
    assert client.sessions == {}


@pytest.mark.parametrize(
    ("list_method", "item_key", "request_method", "value_field"),
    CATALOG_RESPONSES,
)
@pytest.mark.asyncio
async def test_catalogs_omit_first_cursor_forward_exact_cursor_and_preserve_order(
    list_method: str,
    item_key: str,
    request_method: str,
    value_field: str,
) -> None:
    pages = [
        {
            item_key: [_item(item_key, "first"), _item(item_key, "second")],
            "nextCursor": "cursor-exact",
        },
        {item_key: [_item(item_key, "third")]},
    ]
    connection, requests = _scripted_connection(
        lambda index, _method, _params: pages[index - 1]
    )

    expected_values = ["first", "second", "third"]
    if item_key == "resources":
        expected_values = [f"resource:{value}" for value in expected_values]
    assert (
        await _catalog_values(connection, list_method, item_key, value_field)
        == expected_values
    )
    assert requests == [
        (request_method, {}),
        (request_method, {"cursor": "cursor-exact"}),
    ]


@pytest.mark.parametrize(
    ("list_method", "item_key", "request_method", "value_field"),
    CATALOG_RESPONSES,
)
@pytest.mark.asyncio
async def test_catalog_null_cursor_terminates_without_another_request(
    list_method: str,
    item_key: str,
    request_method: str,
    value_field: str,
) -> None:
    connection, requests = _scripted_connection(
        lambda _index, _method, _params: {
            item_key: [_item(item_key, "only")],
            "nextCursor": None,
        }
    )

    expected_value = "resource:only" if item_key == "resources" else "only"
    assert await _catalog_values(connection, list_method, item_key, value_field) == [
        expected_value
    ]
    assert requests == [(request_method, {})]


@pytest.mark.parametrize("cursor", ["", 7, False, [], {}])
@pytest.mark.asyncio
async def test_catalog_rejects_empty_or_non_string_cursor_without_payload_leakage(
    cursor: object,
) -> None:
    sentinel = "private-cursor-payload"
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {
            "tools": [{"name": sentinel}],
            "nextCursor": cursor,
        }
    )

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "Invalid MCP catalog cursor")
    assert sentinel not in str(exc_info.value)
    assert repr(cursor) not in str(exc_info.value)


@pytest.mark.asyncio
async def test_catalog_rejects_repeated_cursor_instead_of_returning_partial_items() -> (
    None
):
    pages = [
        {
            "tools": [{"name": "first", "inputSchema": {"type": "object"}}],
            "nextCursor": "repeat",
        },
        {
            "tools": [{"name": "private-second", "inputSchema": {"type": "object"}}],
            "nextCursor": "repeat",
        },
    ]
    connection, requests = _scripted_connection(
        lambda index, _method, _params: pages[index - 1]
    )

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "Repeated MCP catalog cursor")
    assert "private-second" not in str(exc_info.value)
    assert requests == [
        ("tools/list", {}),
        ("tools/list", {"cursor": "repeat"}),
    ]


@pytest.mark.parametrize("items", [None, "private-items", {"private": "items"}])
@pytest.mark.asyncio
async def test_catalog_rejects_non_list_item_array_without_payload_leakage(
    items: object,
) -> None:
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {"tools": items}
    )

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "Invalid MCP catalog items")
    assert "private" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_catalog_accepts_exactly_100_pages() -> None:
    def respond(index: int, _method: str, _params: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "tools": [{"name": f"tool-{index}", "inputSchema": {"type": "object"}}]
        }
        if index < 100:
            result["nextCursor"] = f"page-{index + 1}"
        return result

    connection, requests = _scripted_connection(respond)

    values = await _catalog_values(connection, "list_tools", "tools", "name")

    assert len(values) == 100
    assert values == [f"tool-{index}" for index in range(1, 101)]
    assert len(requests) == 100
    assert requests[0] == ("tools/list", {})
    assert requests[-1] == ("tools/list", {"cursor": "page-100"})


@pytest.mark.asyncio
async def test_catalog_rejects_page_101_instead_of_returning_100_page_partial() -> None:
    def respond(index: int, _method: str, _params: dict[str, Any]) -> dict[str, Any]:
        return {
            "tools": [{"name": f"tool-{index}", "inputSchema": {"type": "object"}}],
            "nextCursor": f"page-{index + 1}",
        }

    connection, requests = _scripted_connection(respond)

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "MCP catalog page limit exceeded")
    assert len(requests) == 100


@pytest.mark.asyncio
async def test_catalog_accepts_exactly_10_000_items() -> None:
    items = [
        {"name": f"tool-{index}", "inputSchema": {"type": "object"}}
        for index in range(10_000)
    ]
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {"tools": items}
    )

    values = await _catalog_values(connection, "list_tools", "tools", "name")

    assert len(values) == 10_000
    assert values[0] == "tool-0"
    assert values[-1] == "tool-9999"


@pytest.mark.asyncio
async def test_catalog_rejects_item_10_001_instead_of_returning_partial_items() -> None:
    first_page = [
        {"name": f"tool-{index}", "inputSchema": {"type": "object"}}
        for index in range(10_000)
    ]
    pages = [
        {"tools": first_page, "nextCursor": "more"},
        {
            "tools": [
                {
                    "name": "private-item-10001",
                    "inputSchema": {"type": "object"},
                }
            ]
        },
    ]
    connection, requests = _scripted_connection(
        lambda index, _method, _params: pages[index - 1]
    )

    with pytest.raises(Exception) as exc_info:
        await connection.list_tools()

    _assert_client_error(exc_info.value, "MCP catalog item limit exceeded")
    assert "private-item-10001" not in str(exc_info.value)
    assert requests == [
        ("tools/list", {}),
        ("tools/list", {"cursor": "more"}),
    ]


@pytest.mark.parametrize(
    ("list_method", "item_key", "item"),
    [
        pytest.param("list_tools", "tools", {"inputSchema": {}}, id="tool-name"),
        pytest.param(
            "list_tools",
            "tools",
            {"name": "tool", "inputSchema": []},
            id="tool-schema",
        ),
        pytest.param(
            "list_tools",
            "tools",
            {
                "name": "tool",
                "inputSchema": {"type": "object"},
                "annotations": [],
            },
            id="tool-annotations",
        ),
        pytest.param(
            "list_resources", "resources", {"name": "resource"}, id="resource-uri"
        ),
        pytest.param(
            "list_resources",
            "resources",
            {"uri": "resource://1", "name": "resource", "size": True},
            id="resource-size",
        ),
        pytest.param("list_prompts", "prompts", {"arguments": []}, id="prompt-name"),
        pytest.param(
            "list_prompts",
            "prompts",
            {"name": "prompt", "arguments": {}},
            id="prompt-arguments",
        ),
        pytest.param(
            "list_prompts",
            "prompts",
            {"name": "prompt", "arguments": [{"name": "arg", "required": 1}]},
            id="prompt-required",
        ),
        pytest.param(
            "list_prompts",
            "prompts",
            {"name": "x" * 4097, "arguments": []},
            id="bounded-name",
        ),
    ],
)
@pytest.mark.asyncio
async def test_catalog_rejects_invalid_descriptor_shapes_with_fixed_error(
    list_method: str,
    item_key: str,
    item: object,
) -> None:
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {item_key: [item]}
    )

    with pytest.raises(Exception) as exc_info:
        await getattr(connection, list_method)()

    _assert_client_error(exc_info.value, "Invalid MCP catalog items")
    assert "x" * 100 not in str(exc_info.value)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("capabilities", True, id="capabilities-bool"),
        pytest.param("capabilities", [["tools", {}]], id="capabilities-pairs"),
        pytest.param("capabilities", {1: {}}, id="capabilities-key"),
        pytest.param("serverInfo", True, id="server-info-bool"),
        pytest.param("serverInfo", [["name", "server"]], id="server-info-pairs"),
        pytest.param("serverInfo", {1: "server"}, id="server-info-key"),
    ],
)
@pytest.mark.asyncio
async def test_initialize_rejects_nonmapping_or_nonstr_key_metadata(
    field: str,
    value: object,
) -> None:
    connection = _bare_connection()
    connection.client_name = "test-client"

    async def request(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        result = {
            "protocolVersion": "2025-03-26",
            "capabilities": {},
            "serverInfo": {},
        }
        result[field] = value
        return result

    async def notify(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("invalid initialization metadata must not be acknowledged")

    connection.request = request  # type: ignore[method-assign]
    connection.notify = notify  # type: ignore[method-assign]

    with pytest.raises(Exception) as exc_info:
        await connection.initialize()

    _assert_client_error(exc_info.value, "Invalid MCP initialization metadata")


def _invalid_metadata(case: str) -> dict[object, Any]:
    if case == "cycle":
        value: dict[object, Any] = {}
        value["self"] = value
        return value
    if case == "depth":
        value = {}
        current = value
        for _ in range(65):
            child: dict[object, Any] = {}
            current["child"] = child
            current = child
        return value
    if case == "oversize":
        return {"value": "x" * 786_432}
    if case == "non-string-key":
        return {1: "value"}
    if case == "nan":
        return {"value": float("nan")}
    return {"value": float("inf")}


@pytest.mark.parametrize(
    "case", ["cycle", "depth", "oversize", "non-string-key", "nan", "infinity"]
)
@pytest.mark.asyncio
async def test_resource_metadata_rejects_nonfinite_or_unbounded_json(case: str) -> None:
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {
            "contents": [],
            "_meta": _invalid_metadata(case),
        }
    )

    with pytest.raises(Exception) as exc_info:
        await connection.read_resource("note://1")

    _assert_client_error(exc_info.value, "Invalid MCP resource metadata")


@pytest.mark.asyncio
async def test_catalog_preserves_optional_fields_and_detaches_nested_cache() -> None:
    raw_tool = {
        "name": "tool",
        "description": "desc",
        "inputSchema": {"type": "object", "properties": {"x": {"type": "string"}}},
        "annotations": {"audience": ["user"]},
    }
    raw_resource = {
        "uri": "resource://1",
        "name": "resource",
        "description": "desc",
        "mimeType": "text/plain",
        "annotations": {"audience": ["assistant"]},
        "size": 12,
    }
    raw_prompt = {
        "name": "prompt",
        "description": "desc",
        "arguments": [{"name": "topic", "description": "d", "required": True}],
        "annotations": {"audience": ["user"]},
    }
    cached = {}
    for method, key, raw in (
        ("list_tools", "tools", raw_tool),
        ("list_resources", "resources", raw_resource),
        ("list_prompts", "prompts", raw_prompt),
    ):
        connection, _requests = _scripted_connection(
            lambda _index, _request_method, _params, key=key, raw=raw: {key: [raw]}
        )
        cached[key] = getattr(await getattr(connection, method)(), key)

    raw_tool["inputSchema"]["properties"]["x"]["type"] = "number"
    raw_resource["annotations"]["audience"].append("mutated")
    raw_prompt["arguments"][0]["name"] = "mutated"

    client = client_module.MCPClient(name="projection-client")
    client.servers["server"] = cached
    tools = client.get_server_tools("server")
    resources = client.get_server_resources("server")
    prompts = client.get_server_prompts("server")

    assert tools[0]["inputSchema"]["properties"]["x"]["type"] == "string"
    assert tools[0]["annotations"] == {"audience": ["user"]}
    assert resources[0]["annotations"] == {"audience": ["assistant"]}
    assert resources[0]["size"] == 12
    assert prompts[0]["arguments"][0]["name"] == "topic"
    assert prompts[0]["annotations"] == {"audience": ["user"]}

    tools[0]["inputSchema"]["properties"]["x"]["type"] = "integer"
    resources[0]["annotations"]["audience"].append("output")
    prompts[0]["annotations"]["audience"].append("output")
    assert cached["tools"][0].inputSchema["properties"]["x"]["type"] == "string"
    assert cached["resources"][0].annotations == {"audience": ["assistant"]}
    assert cached["prompts"][0].annotations == {"audience": ["user"]}


@pytest.mark.asyncio
async def test_bool_response_id_does_not_consume_integer_request() -> None:
    connection = _bare_connection()
    future = asyncio.get_running_loop().create_future()
    connection._pending_requests = {1: future}

    connection._handle_response({"id": True, "result": {"wrong": True}})

    assert connection._pending_requests == {1: future}
    assert not future.done()
    connection._handle_response({"id": 1, "result": {"right": True}})
    assert future.result() == {"right": True}
    assert connection._pending_requests == {}


def _response_line(total_bytes: int) -> bytes:
    prefix = b'{"jsonrpc":"2.0","id":1,"result":{"value":"'
    suffix = b'"}}\n'
    return prefix + b"x" * (total_bytes - len(prefix) - len(suffix)) + suffix


@pytest.mark.asyncio
async def test_exact_total_output_line_limit_is_accepted() -> None:
    line = _response_line(MAX_OUTPUT_LINE_BYTES)
    assert len(line) == MAX_OUTPUT_LINE_BYTES
    connection = _bare_connection(_Process(_Reader(line, b"")))
    future = asyncio.get_running_loop().create_future()
    connection._pending_requests = {1: future}

    await connection._read_loop()

    assert len(future.result()["value"]) > 1_000_000


@pytest.mark.asyncio
async def test_one_over_total_output_line_limit_is_fatal() -> None:
    line = _response_line(MAX_OUTPUT_LINE_BYTES + 1)
    assert len(line) == MAX_OUTPUT_LINE_BYTES + 1
    connection = _bare_connection(_Process(_Reader(line)))
    future = asyncio.get_running_loop().create_future()
    connection._pending_requests = {1: future}

    await connection._read_loop()

    with pytest.raises(RuntimeError, match="MCP transport unavailable"):
        future.result()
    assert connection._pending_requests == {}


@pytest.mark.parametrize("case", ["missing-reader", "decode", "json", "dispatch"])
@pytest.mark.asyncio
async def test_reader_fatal_paths_fail_pending_and_hide_payload(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "private-reader-payload"
    lines = {
        "decode": b"\xff\n",
        "json": f"{sentinel}\n".encode(),
        "dispatch": b'{"jsonrpc":"2.0","method":"ping","id":9}\n',
    }
    process = _Process(None if case == "missing-reader" else _Reader(lines[case]))
    connection = _bare_connection(process)
    future = asyncio.get_running_loop().create_future()
    connection._pending_requests = {1: future}
    logged: list[object] = []
    monkeypatch.setattr(
        client_module.logger,
        "warning",
        lambda *args, **_kwargs: logged.extend(args),
    )
    if case == "dispatch":

        async def fail_dispatch(_payload: object) -> None:
            raise RuntimeError(sentinel)

        connection._handle_incoming_payload = fail_dispatch  # type: ignore[method-assign]

    await connection._read_loop()

    with pytest.raises(RuntimeError, match="MCP transport unavailable"):
        future.result()
    assert connection._pending_requests == {}
    assert connection._reader_unavailable is True
    assert connection._cleanup_complete is False
    assert sentinel not in str(logged)


@pytest.mark.asyncio
async def test_unrecognized_payload_log_does_not_include_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: list[object] = []
    monkeypatch.setattr(
        client_module.logger,
        "debug",
        lambda *args, **_kwargs: logged.extend(args),
    )
    connection = _bare_connection()

    await connection._handle_incoming_payload({"private": "private-dict-content"})

    assert "private-dict-content" not in str(logged)


@pytest.mark.parametrize("case", ["decode", "json", "oversized", "dispatch"])
@pytest.mark.asyncio
async def test_established_fatal_reader_failure_reaps_child_before_dropping_state(
    case: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "private-fatal-reader"
    lines = {
        "decode": b"\xff\n",
        "json": f"{sentinel}\n".encode(),
        "oversized": _response_line(MAX_OUTPUT_LINE_BYTES + 1),
        "dispatch": b'{"jsonrpc":"2.0","method":"ping","id":9}\n',
    }
    process = _SlowReapProcess(_Reader(lines[case]))
    connection = _bare_connection(process)
    client = client_module.MCPClient(name="fatal-reader-client")
    client.sessions["server"] = connection
    client.servers["server"] = {"command": "fake"}

    async def cleanup() -> None:
        await client._bounded_teardown_connection("server", session=connection)

    connection._on_transport_failure = cleanup
    if case == "dispatch":

        async def fail_dispatch(_payload: object) -> None:
            raise RuntimeError(sentinel)

        connection._handle_incoming_payload = fail_dispatch  # type: ignore[method-assign]

    pending = asyncio.get_running_loop().create_future()
    connection._pending_requests = {1: pending}
    connection._read_task = asyncio.create_task(connection._read_loop())
    try:
        await connection._read_task
        with pytest.raises(RuntimeError, match="MCP transport unavailable"):
            pending.result()
        assert connection._pending_requests == {}

        for _ in range(10):
            if process.wait_started.is_set():
                break
            await asyncio.sleep(0)
        assert process.wait_started.is_set()
        assert client.sessions == {"server": connection}
        assert "server" in client.servers

        cleanup_task = connection._transport_cleanup_task
        assert cleanup_task is not None
        process.allow_reap.set()
        await asyncio.wait_for(cleanup_task, timeout=1)

        assert process.stdin.closed
        assert process.terminate_calls == 1
        assert process.wait_calls == 1
        assert process.returncode == 0
        assert connection._cleanup_complete is True
        assert client.sessions == {}
        assert client.servers == {}
        assert connection._read_task.done()
        assert connection._transport_cleanup_task is None
        await connection.close()
        assert process.terminate_calls == 1
    finally:
        process.allow_reap.set()
        cleanup_task = connection._transport_cleanup_task
        if cleanup_task is not None:
            await asyncio.gather(cleanup_task, return_exceptions=True)
        await connection.close()


@pytest.mark.asyncio
async def test_established_malformed_json_reaps_real_child() -> None:
    script = (
        "import sys,time;"
        "sys.stdin.buffer.readline();"
        "sys.stdout.buffer.write(b'private-malformed-json\\n');"
        "sys.stdout.buffer.flush();"
        "time.sleep(60)"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=MAX_OUTPUT_LINE_BYTES,
    )
    client = client_module.MCPClient(name="real-fatal-reader-client")
    connection: client_module._StdioJSONRPCConnection

    async def cleanup() -> None:
        await client._bounded_teardown_connection("server", session=connection)

    connection = client_module._StdioJSONRPCConnection(
        process,
        client_name="real-fatal-reader-client",
        on_transport_failure=cleanup,
    )
    client.sessions["server"] = connection
    client.servers["server"] = {"command": sys.executable}
    try:
        assert process.stdin is not None
        process.stdin.write(b"go\n")
        await process.stdin.drain()
        await asyncio.wait_for(connection._read_task, timeout=1)
        cleanup_task = connection._transport_cleanup_task
        assert cleanup_task is not None
        await asyncio.wait_for(cleanup_task, timeout=1)

        assert process.returncode is not None
        assert connection._cleanup_complete is True
        assert connection._read_task.done()
        assert connection._stderr_task is None or connection._stderr_task.done()
        assert connection._transport_cleanup_task is None
        assert client.sessions == {}
        assert client.servers == {}
        await connection.close()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()


@pytest.mark.asyncio
async def test_disconnect_cancellation_keeps_recovery_state_until_forced_reap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_started = asyncio.Event()
    process = _SlowReapProcess()

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            close_started.set()
            await asyncio.Future()

    session = Session()
    client = client_module.MCPClient(name="disconnect-cancel-client")
    client.sessions["server"] = session  # type: ignore[assignment]
    client.servers["server"] = {"command": "fake"}
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    task = asyncio.create_task(client.disconnect_from_server("server"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    state_retained_during_cleanup = (
        not task.done()
        and client.sessions.get("server") is session
        and "server" in client.servers
    )

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert state_retained_during_cleanup
    assert process.stdin.closed
    assert process.returncode is not None
    assert process.kill_calls == 1
    assert process.wait_calls == 2
    assert client.sessions == {}
    assert client.servers == {}
    assert await client.disconnect_from_server("server") is False


@pytest.mark.asyncio
async def test_disconnect_all_cancellation_still_reaps_every_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = asyncio.Event()
    processes = [_SlowReapProcess(), _SlowReapProcess()]
    close_calls: list[str] = []

    class Session:
        def __init__(self, server_id: str, process: _SlowReapProcess) -> None:
            self.server_id = server_id
            self.process = process

        async def close(self) -> None:
            close_calls.append(self.server_id)
            if self.server_id == "first":
                first_started.set()
                await asyncio.Future()
            self.process.stdin.close()
            self.process.terminate()
            self.process.allow_reap.set()
            await self.process.wait()

    client = client_module.MCPClient(name="disconnect-all-cancel-client")
    for server_id, process in zip(("first", "second"), processes):
        client.sessions[server_id] = Session(server_id, process)  # type: ignore[assignment]
        client.servers[server_id] = {"command": "fake"}
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    task = asyncio.create_task(client.disconnect_all())
    await asyncio.wait_for(first_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)

    assert close_calls == ["first", "second"]
    assert all(process.returncode is not None for process in processes)
    assert client.sessions == {}
    assert client.servers == {}


@pytest.mark.asyncio
async def test_notification_log_is_fixed_and_payload_free(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = "private-notification-method"
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(
        client_module.logger,
        "debug",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    connection = _bare_connection()

    await connection._handle_incoming_payload({"method": sentinel})

    captured = capsys.readouterr()
    assert calls == [(("Ignoring MCP server notification",), {})]
    assert sentinel not in repr(calls)
    assert sentinel not in captured.out
    assert sentinel not in captured.err


@pytest.mark.parametrize(
    ("list_method", "item_key", "item"),
    [
        pytest.param(
            "list_tools",
            "tools",
            {"name": "bad name", "inputSchema": {"type": "object"}},
            id="tool-name-pattern",
        ),
        pytest.param(
            "list_tools",
            "tools",
            {"name": "tool", "inputSchema": {}},
            id="tool-schema-root-missing",
        ),
        pytest.param(
            "list_tools",
            "tools",
            {"name": "tool", "inputSchema": {"type": "array"}},
            id="tool-schema-root-wrong",
        ),
        pytest.param(
            "list_tools",
            "tools",
            {
                "name": "tool",
                "inputSchema": {"type": "object"},
                "annotations": {"audience": ["private-role"]},
            },
            id="annotation-audience",
        ),
        pytest.param(
            "list_tools",
            "tools",
            {
                "name": "tool",
                "inputSchema": {"type": "object"},
                "annotations": {"priority": True},
            },
            id="annotation-priority-bool",
        ),
        pytest.param(
            "list_tools",
            "tools",
            {
                "name": "tool",
                "inputSchema": {"type": "object"},
                "annotations": {"priority": 1.1},
            },
            id="annotation-priority-range",
        ),
        pytest.param(
            "list_tools",
            "tools",
            {
                "name": "tool",
                "inputSchema": {"type": "object"},
                "annotations": {"readOnlyHint": "yes"},
            },
            id="tool-annotation-hint",
        ),
        pytest.param(
            "list_resources",
            "resources",
            {"name": "resource", "uri": "relative/path"},
            id="resource-relative-uri",
        ),
        pytest.param(
            "list_resources",
            "resources",
            {"name": "resource", "uri": "https:///missing-host"},
            id="resource-http-host",
        ),
        pytest.param(
            "list_resources",
            "resources",
            {"name": "resource", "uri": "https://user@example.test/private"},
            id="resource-credentials",
        ),
        pytest.param(
            "list_resources",
            "resources",
            {"name": "x" * 513, "uri": "resource://valid"},
            id="resource-name-bound",
        ),
        pytest.param(
            "list_resources",
            "resources",
            {
                "name": "resource",
                "uri": "resource://valid",
                "mimeType": "x" * 256,
            },
            id="resource-mime-bound",
        ),
        pytest.param(
            "list_prompts",
            "prompts",
            {"name": "bad name", "arguments": []},
            id="prompt-name-pattern",
        ),
        pytest.param(
            "list_prompts",
            "prompts",
            {
                "name": "prompt",
                "arguments": [{"name": "same"}, {"name": "same"}],
            },
            id="prompt-duplicate-argument",
        ),
        pytest.param(
            "list_prompts",
            "prompts",
            {"name": "prompt", "arguments": [{"name": "x" * 129}]},
            id="prompt-argument-bound",
        ),
    ],
)
@pytest.mark.asyncio
async def test_catalog_rejects_profile_invalid_descriptors_with_fixed_error(
    list_method: str,
    item_key: str,
    item: dict[str, object],
) -> None:
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {item_key: [item]}
    )

    with pytest.raises(Exception) as exc_info:
        await getattr(connection, list_method)()

    _assert_client_error(exc_info.value, "Invalid MCP catalog items")
    assert "private" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_catalog_accepts_exact_profile_descriptor_controls() -> None:
    items = {
        "tools": [
            {
                "name": "tool.one-2",
                "inputSchema": {"type": "object"},
                "annotations": {
                    "audience": ["user", "assistant"],
                    "priority": 0.5,
                    "title": "Tool title",
                    "readOnlyHint": True,
                    "destructiveHint": False,
                    "idempotentHint": True,
                    "openWorldHint": False,
                },
            }
        ],
        "resources": [
            {
                "name": "Resource display name",
                "uri": "custom://Host/path?q=1",
                "mimeType": "text/plain",
                "annotations": {"audience": ["assistant"], "priority": 1},
            }
        ],
        "prompts": [
            {
                "name": "prompt.one-2",
                "arguments": [
                    {"name": "first argument", "required": True},
                    {"name": "second", "required": False},
                ],
                "annotations": {"audience": ["user"], "priority": 0},
            }
        ],
    }

    projected: dict[str, SimpleNamespace] = {}
    for list_method, item_key in (
        ("list_tools", "tools"),
        ("list_resources", "resources"),
        ("list_prompts", "prompts"),
    ):
        connection, _requests = _scripted_connection(
            lambda _index, _method, _params, item_key=item_key: {
                item_key: items[item_key]
            }
        )
        result = await getattr(connection, list_method)()
        assert len(getattr(result, item_key)) == 1
        projected[item_key] = getattr(result, item_key)[0]

    assert vars(projected["tools"]) == {
        "name": "tool.one-2",
        "description": "",
        "inputSchema": {"type": "object"},
        "annotations": items["tools"][0]["annotations"],
    }
    assert vars(projected["resources"]) == {
        "uri": "custom://Host/path?q=1",
        "name": "Resource display name",
        "description": "",
        "mimeType": "text/plain",
        "annotations": items["resources"][0]["annotations"],
        "size": None,
    }
    assert vars(projected["prompts"]) == {
        "name": "prompt.one-2",
        "description": "",
        "arguments": projected["prompts"].arguments,
        "annotations": items["prompts"][0]["annotations"],
    }
    assert [vars(argument) for argument in projected["prompts"].arguments] == [
        {"name": "first argument", "description": "", "required": True},
        {"name": "second", "description": "", "required": False},
    ]


@pytest.mark.asyncio
async def test_low_level_resource_read_copies_exact_result_metadata() -> None:
    metadata = {
        "tldw.chatbook/continuation": {"hasMore": True, "nextUri": "note://2"},
        "tldw.chatbook/resource": {"kind": "note"},
    }
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {
            "contents": [{"uri": "note://1", "mimeType": "text/plain", "text": "body"}],
            "_meta": metadata,
        }
    )

    result = await connection.read_resource("note://1")

    assert result._meta == metadata
    assert result._meta is not metadata
    metadata["tldw.chatbook/continuation"]["hasMore"] = False
    assert result._meta["tldw.chatbook/continuation"]["hasMore"] is True
    metadata["late-mutation"] = True
    assert "late-mutation" not in result._meta


@pytest.mark.parametrize("metadata", [None, pytest.param("absent", id="absent")])
@pytest.mark.asyncio
async def test_low_level_resource_read_defaults_missing_or_null_metadata_to_empty(
    metadata: object,
) -> None:
    payload: dict[str, Any] = {"contents": []}
    if metadata is None:
        payload["_meta"] = None
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: payload
    )

    result = await connection.read_resource("note://1")

    assert result._meta == {}


@pytest.mark.parametrize("metadata", ["private-metadata", [], 7, True])
@pytest.mark.asyncio
async def test_low_level_resource_read_rejects_invalid_metadata_without_payload_leakage(
    metadata: object,
) -> None:
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {"contents": [], "_meta": metadata}
    )

    with pytest.raises(Exception) as exc_info:
        await connection.read_resource("note://1")

    _assert_client_error(exc_info.value, "Invalid MCP resource metadata")
    assert "private-metadata" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_high_level_resource_read_preserves_exact_metadata_key_and_copies_it() -> (
    None
):
    metadata = {
        "tldw.chatbook/continuation": {"hasMore": False, "nextUri": None},
        "tldw.chatbook/resource": {"kind": "note"},
    }

    class Session:
        async def read_resource(self, resource_uri: str) -> SimpleNamespace:
            assert resource_uri == "note://1"
            return SimpleNamespace(
                contents=[SimpleNamespace(text="body", mimeType="text/markdown")],
                _meta=metadata,
            )

    client = client_module.MCPClient.__new__(client_module.MCPClient)
    client.sessions = {"server": Session()}  # type: ignore[dict-item]

    result = await client.read_resource("server", "note://1")

    assert result == {
        "uri": "note://1",
        "content": "body",
        "mimeType": "text/markdown",
        "_meta": metadata,
    }
    assert result["_meta"] is not metadata
    result["_meta"]["tldw.chatbook/continuation"]["hasMore"] = True
    assert metadata["tldw.chatbook/continuation"]["hasMore"] is False
    result["_meta"]["late-mutation"] = True
    assert "late-mutation" not in metadata


@pytest.mark.asyncio
async def test_high_level_resource_read_rejects_invalid_metadata_without_payload_leakage() -> (
    None
):
    class Session:
        async def read_resource(self, resource_uri: str) -> SimpleNamespace:
            assert resource_uri == "note://1"
            return SimpleNamespace(contents=[], _meta="private-metadata")

    client = client_module.MCPClient.__new__(client_module.MCPClient)
    client.sessions = {"server": Session()}  # type: ignore[dict-item]

    result = await client.read_resource("server", "note://1")

    assert result == {"error": "Invalid MCP resource metadata"}
    assert "private-metadata" not in str(result)
