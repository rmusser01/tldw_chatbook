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


class _RetryAfterPermissionErrorProcess(_Process):
    def __init__(self, sentinel: str) -> None:
        super().__init__()
        self.sentinel = sentinel

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        if self.kill_calls == 1:
            raise PermissionError(self.sentinel)
        self.returncode = -9

    async def wait(self) -> int:
        self.wait_calls += 1
        if self.returncode is None:
            await asyncio.Future()
        return self.returncode


class _ConnectSession:
    protocol_version = "2025-03-26"
    server_info: dict[str, Any] = {}
    server_capabilities: dict[str, Any] = {}

    def __init__(self, process: _Process, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
        self.process = process
        self._on_transport_failure: Callable[[], Any] | None = None

    async def initialize(self) -> None:
        return None

    async def close(self) -> None:
        self.process.terminate()
        await self.process.wait()


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


@pytest.mark.asyncio
async def test_init_failure_retains_private_owner_until_retry_reaps_child(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _RetryAfterPermissionErrorProcess("private-init-kill-payload")
    created_session: object | None = None

    class Session:
        def __init__(self, created_process: object, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
            nonlocal created_session
            assert created_process is process
            assert client_name == "pending-init-client"
            self.process = created_process
            created_session = self

        async def initialize(self) -> None:
            raise RuntimeError("initialization rejected")

        async def close(self) -> None:
            await asyncio.Future()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    client = client_module.MCPClient(name="pending-init-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    assert await client.connect_to_server("server", "python") is False
    assert client.sessions == {}
    assert client.servers == {}
    pending = client._pending_connections["server"]
    assert pending.process is process
    assert pending.session is created_session
    assert process.returncode is None
    assert client._connect_reservations == {}

    assert await client.disconnect_from_server("server") is True
    assert process.returncode == -9
    assert process.kill_calls == 2
    assert client._pending_connections == {}
    assert client.sessions == {}
    assert client.servers == {}
    await asyncio.sleep(0)
    current = asyncio.current_task()
    assert all(
        task is current
        or task.done()
        or "_finish_connection_cleanup" not in task.get_coro().__qualname__
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_discovery_does_not_publish_half_initialized_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_started = asyncio.Event()
    allow_discovery = asyncio.Event()
    process = _Process()

    class Session:
        protocol_version = "2025-03-26"
        server_info: dict[str, Any] = {}
        server_capabilities: dict[str, Any] = {}

        def __init__(self, created_process: object, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
            assert created_process is process
            self.process = created_process

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            process.terminate()
            await process.wait()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    async def discover(_server_id: str) -> None:
        discovery_started.set()
        await allow_discovery.wait()

    client = client_module.MCPClient(name="atomic-publication-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)

    connect = asyncio.create_task(client.connect_to_server("server", "python"))
    await asyncio.wait_for(discovery_started.wait(), timeout=1)
    try:
        assert client.sessions == {}
        assert client.servers == {}
        assert client._pending_connections["server"].session is not None
        allow_discovery.set()
        assert await asyncio.wait_for(connect, timeout=1) is True
        assert "server" in client.sessions
        assert "server" in client.servers
        assert client._pending_connections == {}
    finally:
        allow_discovery.set()
        if not connect.done():
            connect.cancel()
            await asyncio.gather(connect, return_exceptions=True)
        await client.disconnect_all()


@pytest.mark.asyncio
async def test_pending_cleanup_preserves_registry_replacement_identity() -> None:
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    process = _Process()

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            close_started.set()
            await allow_close.wait()
            process.terminate()
            await process.wait()

    old_session = Session()
    old_pending = SimpleNamespace(process=process, session=old_session)
    replacement = SimpleNamespace(process=_Process(), session=None)
    client = client_module.MCPClient(name="pending-replacement-client")
    client._pending_connections["server"] = old_pending

    cleanup = asyncio.create_task(client.disconnect_from_server("server"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    client._pending_connections["server"] = replacement
    allow_close.set()

    assert await asyncio.wait_for(cleanup, timeout=1) is True
    assert process.returncode == 0
    assert client._pending_connections == {"server": replacement}
    assert client.sessions == {}
    assert client.servers == {}


@pytest.mark.asyncio
async def test_spawn_only_pending_cleanup_does_not_touch_active_replacement() -> None:
    old_process = _Process()
    replacement_process = _Process()

    class ReplacementSession:
        def __init__(self) -> None:
            self.process = replacement_process
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1
            replacement_process.terminate()

    old_pending = SimpleNamespace(process=old_process, session=None)
    replacement = ReplacementSession()
    replacement_record = {"command": "replacement"}
    client = client_module.MCPClient(name="spawn-replacement-client")
    client._pending_connections["server"] = old_pending
    client.sessions["server"] = replacement  # type: ignore[assignment]
    client.servers["server"] = replacement_record

    await client._bounded_teardown_connection("server", pending=old_pending)

    assert old_process.returncode == 0
    assert replacement.close_calls == 0
    assert replacement_process.returncode is None
    assert client.sessions == {"server": replacement}
    assert client.servers == {"server": replacement_record}
    assert client._pending_connections == {}


@pytest.mark.asyncio
async def test_pending_collision_cleanup_failure_does_not_spawn_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _RetryAfterPermissionErrorProcess("private-collision-payload")

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            await asyncio.Future()

    session = Session()
    pending = SimpleNamespace(process=process, session=session)
    client = client_module.MCPClient(name="pending-collision-client")
    client._pending_connections["server"] = pending
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    async def unexpected_spawn(*_args: Any, **_kwargs: Any) -> _Process:
        pytest.fail("replacement must not spawn while the prior child remains live")

    monkeypatch.setattr(asyncio, "create_subprocess_exec", unexpected_spawn)

    assert await client.connect_to_server("server", "python") is False
    assert client._pending_connections == {"server": pending}
    assert process.returncode is None
    assert client._connect_reservations == {}
    assert await client.disconnect_from_server("server") is True
    assert process.returncode == -9
    assert client._pending_connections == {}


@pytest.mark.asyncio
async def test_pending_transport_failure_reaps_without_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _Process()

    class Session:
        protocol_version = "2025-03-26"
        server_info: dict[str, Any] = {}
        server_capabilities: dict[str, Any] = {}

        def __init__(self, created_process: object, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
            assert created_process is process
            self.process = created_process
            self._on_transport_failure: Callable[[], Any] | None = None

        async def initialize(self) -> None:
            assert self._on_transport_failure is not None
            await self._on_transport_failure()
            raise RuntimeError("transport failed")

        async def close(self) -> None:
            process.terminate()
            await process.wait()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    client = client_module.MCPClient(name="pending-transport-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)

    assert await client.connect_to_server("server", "python") is False
    assert process.returncode == 0
    assert client.sessions == {}
    assert client.servers == {}
    assert client._pending_connections == {}
    await asyncio.sleep(0)
    current = asyncio.current_task()
    assert all(
        task is current
        or task.done()
        or "cleanup_failed_transport" not in task.get_coro().__qualname__
        for task in asyncio.all_tasks()
    )


@pytest.mark.asyncio
async def test_same_id_connect_reserves_before_spawn_await(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_spawn_started = asyncio.Event()
    allow_first_spawn = asyncio.Event()
    processes: list[_Process] = []

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        process = _Process()
        processes.append(process)
        if len(processes) == 1:
            first_spawn_started.set()
            await allow_first_spawn.wait()
        return process

    async def discover(_server_id: str) -> None:
        return None

    client = client_module.MCPClient(name="same-id-reservation-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", _ConnectSession)
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)

    first = asyncio.create_task(client.connect_to_server("server", "python"))
    await asyncio.wait_for(first_spawn_started.wait(), timeout=1)
    try:
        second_result = await asyncio.wait_for(
            client.connect_to_server("server", "python"), timeout=1
        )
        allow_first_spawn.set()
        first_result = await asyncio.wait_for(first, timeout=1)

        assert first_result is True
        assert second_result is False
        assert len(processes) == 1
        assert client.sessions["server"].process is processes[0]
        assert client._pending_connections == {}
        assert getattr(client, "_connect_reservations", {}) == {}
    finally:
        allow_first_spawn.set()
        if not first.done():
            first.cancel()
            await asyncio.gather(first, return_exceptions=True)
        await client.disconnect_all()
        for process in processes:
            if process.returncode is None:
                process.kill()
                await process.wait()


@pytest.mark.asyncio
async def test_same_id_collision_during_failed_init_retains_retryable_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_started = asyncio.Event()
    allow_spawn = asyncio.Event()
    processes: list[_RetryAfterPermissionErrorProcess] = []

    async def spawn(*_args: Any, **_kwargs: Any) -> _RetryAfterPermissionErrorProcess:
        process = _RetryAfterPermissionErrorProcess(
            f"private-reserved-child-{len(processes)}"
        )
        processes.append(process)
        if len(processes) == 1:
            spawn_started.set()
            await allow_spawn.wait()
        return process

    class Session:
        def __init__(self, process: _Process, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
            self.process = process

        async def initialize(self) -> None:
            raise RuntimeError("private-initialization-payload")

        async def close(self) -> None:
            await asyncio.Future()

    client = client_module.MCPClient(name="same-id-failed-init-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    first = asyncio.create_task(client.connect_to_server("server", "python"))
    await asyncio.wait_for(spawn_started.wait(), timeout=1)
    assert await client.connect_to_server("server", "python") is False
    allow_spawn.set()
    assert await asyncio.wait_for(first, timeout=1) is False

    assert len(processes) == 1
    assert processes[0].returncode is None
    assert client.sessions == {}
    assert client.servers == {}
    assert client._pending_connections["server"].process is processes[0]
    assert client._connect_reservations == {}

    assert await client.disconnect_from_server("server") is True
    assert processes[0].returncode == -9
    assert processes[0].kill_calls == 2
    assert client._pending_connections == {}


@pytest.mark.parametrize("owner_kind", ["active", "pending"])
@pytest.mark.asyncio
async def test_spawn_resume_does_not_replace_another_owner(
    owner_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_started = asyncio.Event()
    allow_spawn = asyncio.Event()
    child = _Process()
    replacement_process = _Process()
    replacement_session = SimpleNamespace(process=replacement_process)
    replacement_record = {"command": "replacement"}

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        spawn_started.set()
        await allow_spawn.wait()
        return child

    async def discover(_server_id: str) -> None:
        return None

    client = client_module.MCPClient(name="resume-owner-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", _ConnectSession)
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)

    connect = asyncio.create_task(client.connect_to_server("server", "python"))
    await asyncio.wait_for(spawn_started.wait(), timeout=1)
    if owner_kind == "active":
        client.sessions["server"] = replacement_session  # type: ignore[assignment]
        client.servers["server"] = replacement_record
    else:
        replacement = SimpleNamespace(
            process=replacement_process, session=replacement_session
        )
        client._pending_connections["server"] = replacement
    allow_spawn.set()

    assert await asyncio.wait_for(connect, timeout=1) is False
    assert child.returncode == 0
    if owner_kind == "active":
        assert client.sessions == {"server": replacement_session}
        assert client.servers == {"server": replacement_record}
        assert client._pending_connections == {}
    else:
        assert client.sessions == {}
        assert client.servers == {}
        assert client._pending_connections == {"server": replacement}
    assert getattr(client, "_connect_reservations", {}) == {}


@pytest.mark.asyncio
async def test_connect_releases_only_its_own_reservation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spawn_started = asyncio.Event()
    allow_spawn = asyncio.Event()
    child = _Process()
    replacement_reservation = object()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        spawn_started.set()
        await allow_spawn.wait()
        return child

    client = client_module.MCPClient(name="reservation-identity-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", _ConnectSession)

    connect = asyncio.create_task(client.connect_to_server("server", "python"))
    await asyncio.wait_for(spawn_started.wait(), timeout=1)
    client._connect_reservations = {"server": replacement_reservation}
    allow_spawn.set()

    assert await asyncio.wait_for(connect, timeout=1) is False
    assert child.returncode == 0
    assert client.sessions == {}
    assert client.servers == {}
    assert client._pending_connections == {}
    assert client._connect_reservations == {"server": replacement_reservation}
    client._connect_reservations.pop("server")


@pytest.mark.asyncio
async def test_different_server_ids_can_spawn_concurrently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    both_spawns_started = asyncio.Event()
    allow_spawns = asyncio.Event()
    processes: dict[str, _Process] = {}

    async def spawn(command: str, *_args: Any, **_kwargs: Any) -> _Process:
        process = _Process()
        processes[command] = process
        if len(processes) == 2:
            both_spawns_started.set()
        await allow_spawns.wait()
        return process

    async def discover(_server_id: str) -> None:
        return None

    client = client_module.MCPClient(name="different-id-reservation-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", _ConnectSession)
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)

    first = asyncio.create_task(client.connect_to_server("first", "first-command"))
    second = asyncio.create_task(client.connect_to_server("second", "second-command"))
    await asyncio.wait_for(both_spawns_started.wait(), timeout=1)
    allow_spawns.set()
    try:
        assert await asyncio.gather(first, second) == [True, True]
        assert set(client.sessions) == {"first", "second"}
        assert getattr(client, "_connect_reservations", {}) == {}
    finally:
        allow_spawns.set()
        await client.disconnect_all()


@pytest.mark.parametrize("phase", ["initialize", "discovery"])
@pytest.mark.asyncio
async def test_connect_failure_log_is_fixed_and_payload_free(
    phase: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    sentinel = f"private-{phase}-jsonrpc-payload"
    process = _Process()
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def failure() -> client_module._JSONRPCError:
        return client_module._JSONRPCError(
            {
                "code": -32000,
                "message": sentinel,
                "data": {"private": sentinel},
            }
        )

    class Session(_ConnectSession):
        async def initialize(self) -> None:
            if phase == "initialize":
                raise failure()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    async def discover(_server_id: str) -> None:
        if phase == "discovery":
            raise failure()

    client = client_module.MCPClient(name="failure-log-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)
    monkeypatch.setattr(client, "_discover_server_capabilities", discover)
    monkeypatch.setattr(
        client_module.logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    assert await client.connect_to_server("server", "python") is False

    captured = capsys.readouterr()
    assert logged == [(("Failed to connect to MCP server",), {})]
    assert sentinel not in repr(logged)
    assert sentinel not in captured.out
    assert sentinel not in captured.err
    assert process.returncode == 0
    assert client.sessions == {}
    assert client.servers == {}
    assert client._pending_connections == {}


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

        def __init__(self, created_process: object, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
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
    assert client._connect_reservations == {}


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

        def __init__(self, created_process: object, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
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
    assert client._connect_reservations == {}


@pytest.mark.parametrize(
    (
        "failure_stage",
        "expected_message",
        "expected_wait_calls",
        "expected_kills",
        "expected_success",
    ),
    [
        pytest.param(
            "stdin-close",
            "Failed to close MCP subprocess stdin during forced cleanup",
            1,
            0,
            True,
            id="stdin-close",
        ),
        pytest.param(
            "initial-wait",
            "Failed to wait for MCP subprocess termination during forced cleanup",
            2,
            1,
            True,
            id="initial-wait",
        ),
        pytest.param(
            "final-reap",
            "Failed to reap MCP subprocess after forced cleanup",
            2,
            1,
            False,
            id="final-reap",
        ),
    ],
)
@pytest.mark.asyncio
async def test_forced_cleanup_reports_failures_and_only_finalizes_reaped_registry(
    failure_stage: str,
    expected_message: str,
    expected_wait_calls: int,
    expected_kills: int,
    expected_success: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "private-cleanup-payload"
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FailingStdin(_Stdin):
        def close(self) -> None:
            self.closed = True
            if failure_stage == "stdin-close":
                raise RuntimeError(sentinel)

    class FailingProcess(_Process):
        def __init__(self) -> None:
            super().__init__()
            self.stdin = FailingStdin()

        def terminate(self) -> None:
            self.terminate_calls += 1
            if failure_stage == "stdin-close":
                self.returncode = 0

        async def wait(self) -> int:
            self.wait_calls += 1
            if failure_stage == "initial-wait" and self.wait_calls == 1:
                raise RuntimeError(sentinel)
            if failure_stage == "final-reap":
                if self.wait_calls == 1:
                    await asyncio.Future()
                raise RuntimeError(sentinel)
            return self.returncode or 0

    process = FailingProcess()

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            await asyncio.Future()

    session = Session()
    client = client_module.MCPClient(name="forced-cleanup-client")
    client.sessions["server"] = session  # type: ignore[assignment]
    client.servers["server"] = {"command": "fake"}
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        client_module.logger,
        "warning",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    if expected_success:
        await asyncio.wait_for(client._bounded_teardown_connection("server"), timeout=1)
    else:
        with pytest.raises(client_module.MCPClientError) as exc_info:
            await asyncio.wait_for(
                client._bounded_teardown_connection("server"), timeout=1
            )
        assert str(exc_info.value) == "MCP subprocess cleanup incomplete"

    assert logged == [((expected_message,), {})]
    assert sentinel not in repr(logged)
    assert process.stdin.closed
    assert process.terminate_calls == 1
    assert process.wait_calls == expected_wait_calls
    assert process.kill_calls == expected_kills
    if expected_success:
        assert client.sessions == {}
        assert client.servers == {}
    else:
        assert client.sessions == {"server": session}
        assert client.servers == {"server": {"command": "fake"}}


@pytest.mark.parametrize(
    "wait_delivery_delay",
    [
        pytest.param(0.0, id="immediate"),
        pytest.param(0.02, id="delayed"),
    ],
)
@pytest.mark.asyncio
async def test_disconnect_retains_live_real_child_after_kill_permission_error_and_retries(
    monkeypatch: pytest.MonkeyPatch,
    wait_delivery_delay: float,
) -> None:
    child = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        "import time; time.sleep(60)",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    sentinel = "private-kill-permission-payload"

    class Process:
        def __init__(self) -> None:
            self.stdin = child.stdin
            self.kill_calls = 0
            self.kill_succeeded = False

        @property
        def returncode(self) -> int | None:
            return child.returncode

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            self.kill_calls += 1
            if self.kill_calls == 1:
                raise PermissionError(sentinel)
            child.kill()
            self.kill_succeeded = True

        async def wait(self) -> int:
            returncode = await child.wait()
            if self.kill_succeeded:
                await asyncio.sleep(wait_delivery_delay)
            return returncode

    process = Process()

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            await asyncio.Future()

    session = Session()
    client = client_module.MCPClient(name="kill-permission-client")
    server_record = {"command": sys.executable}
    client.sessions["server"] = session  # type: ignore[assignment]
    client.servers["server"] = server_record
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.25)
    monkeypatch.setattr(
        client_module.logger,
        "error",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )
    try:
        assert await client.disconnect_from_server("server") is False
        assert child.returncode is None
        assert client.sessions == {"server": session}
        assert client.servers == {"server": server_record}
        assert sentinel not in repr(logged)

        assert await client.disconnect_from_server("server") is True
        assert child.returncode is not None
        assert process.kill_calls == 2
        assert client.sessions == {}
        assert client.servers == {}
    finally:
        if child.returncode is None:
            child.kill()
            await child.wait()


@pytest.mark.asyncio
async def test_failed_old_session_cleanup_preserves_registry_replacement_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_started = asyncio.Event()

    class Process(_Process):
        def terminate(self) -> None:
            self.terminate_calls += 1

        def kill(self) -> None:
            self.kill_calls += 1
            raise PermissionError("private-replacement-payload")

        async def wait(self) -> int:
            self.wait_calls += 1
            await asyncio.Future()

    process = Process()

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            close_started.set()
            await asyncio.Future()

    old_session = Session()
    replacement = _bare_connection()
    replacement_record = {"command": "replacement"}
    client = client_module.MCPClient(name="replacement-identity-client")
    client.sessions["server"] = old_session  # type: ignore[assignment]
    client.servers["server"] = {"command": "old"}
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    cleanup = asyncio.create_task(client.disconnect_from_server("server"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    client.sessions["server"] = replacement
    client.servers["server"] = replacement_record

    assert await asyncio.wait_for(cleanup, timeout=1) is False
    assert process.returncode is None
    assert client.sessions == {"server": replacement}
    assert client.servers == {"server": replacement_record}


@pytest.mark.asyncio
async def test_successful_old_session_cleanup_preserves_registry_replacement_identity() -> (
    None
):
    close_started = asyncio.Event()
    allow_close = asyncio.Event()
    process = _Process()

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            close_started.set()
            await allow_close.wait()
            process.terminate()
            await process.wait()

    old_session = Session()
    replacement = _bare_connection()
    replacement_record = {"command": "replacement"}
    client = client_module.MCPClient(name="replacement-success-client")
    client.sessions["server"] = old_session  # type: ignore[assignment]
    client.servers["server"] = {"command": "old"}

    cleanup = asyncio.create_task(client.disconnect_from_server("server"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    client.sessions["server"] = replacement
    client.servers["server"] = replacement_record
    allow_close.set()

    assert await asyncio.wait_for(cleanup, timeout=1) is True
    assert process.returncode == 0
    assert client.sessions == {"server": replacement}
    assert client.servers == {"server": replacement_record}


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


@pytest.mark.parametrize(
    (
        "failure_stage",
        "expected_message",
        "expected_waits",
        "expected_kills",
        "expected_returncode",
    ),
    [
        pytest.param(
            "stdin-close",
            "Failed to close MCP subprocess stdin",
            1,
            0,
            0,
            id="stdin-close",
        ),
        pytest.param(
            "stdin-wait",
            "Failed to wait for MCP subprocess stdin closure",
            1,
            0,
            0,
            id="stdin-wait",
        ),
        pytest.param(
            "terminate",
            "Failed to terminate MCP subprocess cleanly",
            1,
            1,
            -9,
            id="terminate",
        ),
        pytest.param(
            "initial-wait",
            "Failed to wait for MCP subprocess termination",
            2,
            1,
            -9,
            id="initial-wait",
        ),
        pytest.param(
            "final-reap",
            "Failed to reap MCP subprocess after kill",
            2,
            1,
            -9,
            id="final-reap",
        ),
        pytest.param(
            "kill",
            "Failed to kill MCP subprocess cleanly",
            1,
            1,
            None,
            id="kill",
        ),
        pytest.param(
            "done-task",
            "MCP transport task failed before cleanup",
            1,
            0,
            0,
            id="done-task",
        ),
        pytest.param(
            "active-task",
            "MCP transport task failed during cleanup",
            1,
            0,
            0,
            id="active-task",
        ),
    ],
)
@pytest.mark.asyncio
async def test_connection_close_reports_failures_and_finishes_cleanup(
    failure_stage: str,
    expected_message: str,
    expected_waits: int,
    expected_kills: int,
    expected_returncode: int | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "private-close-payload"
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class FailingStdin(_Stdin):
        def close(self) -> None:
            self.closed = True
            if failure_stage == "stdin-close":
                raise RuntimeError(sentinel)

        async def wait_closed(self) -> None:
            if failure_stage == "stdin-wait":
                raise RuntimeError(sentinel)

    class FailingProcess(_Process):
        def __init__(self) -> None:
            super().__init__()
            self.stdin = FailingStdin()

        def terminate(self) -> None:
            self.terminate_calls += 1
            if failure_stage == "terminate":
                raise RuntimeError(sentinel)
            if failure_stage not in {"initial-wait", "final-reap", "kill"}:
                self.returncode = 0

        def kill(self) -> None:
            self.kill_calls += 1
            if failure_stage == "kill":
                raise RuntimeError(sentinel)
            self.returncode = -9

        async def wait(self) -> int:
            self.wait_calls += 1
            if failure_stage == "initial-wait" and self.wait_calls == 1:
                raise RuntimeError(sentinel)
            if failure_stage in {"final-reap", "kill"} and self.wait_calls == 1:
                await asyncio.Future()
            if failure_stage == "final-reap":
                raise RuntimeError(sentinel)
            return self.returncode or 0

    process = FailingProcess()
    connection = _bare_connection(process)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        client_module.logger,
        "warning",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    if failure_stage == "done-task":

        async def fail_before_cleanup() -> None:
            raise RuntimeError(sentinel)

        connection._read_task = asyncio.create_task(fail_before_cleanup())
        await asyncio.sleep(0)
    elif failure_stage == "active-task":
        started = asyncio.Event()

        async def fail_during_cleanup() -> None:
            started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                raise RuntimeError(sentinel) from None

        connection._read_task = asyncio.create_task(fail_during_cleanup())
        await asyncio.wait_for(started.wait(), timeout=1)

    await asyncio.wait_for(connection.close(), timeout=1)

    assert logged == [((expected_message,), {})]
    assert sentinel not in repr(logged)
    assert process.stdin.closed
    assert process.terminate_calls == 1
    assert process.wait_calls == expected_waits
    assert process.kill_calls == expected_kills
    assert process.returncode == expected_returncode
    assert connection._cleanup_complete is True


@pytest.mark.asyncio
async def test_teardown_close_failure_is_reported_reaped_and_finalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "private-session-close-payload"
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []
    process = _Process()

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            raise RuntimeError(sentinel)

    session = Session()
    client = client_module.MCPClient(name="failing-close-client")
    client.sessions["server"] = session  # type: ignore[assignment]
    client.servers["server"] = {"command": "fake"}
    monkeypatch.setattr(
        client_module.logger,
        "warning",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    await asyncio.wait_for(client._bounded_teardown_connection("server"), timeout=1)

    assert logged == [(("Failed to close MCP connection during teardown",), {})]
    assert sentinel not in repr(logged)
    assert process.returncode == 0
    assert process.wait_calls == 1
    assert client.sessions == {}
    assert client.servers == {}


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

        def __init__(self, created_process: object, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
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
    assert client._connect_reservations == {}


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


@pytest.mark.parametrize("reader", [None, _Reader(b"")], ids=["missing-stdout", "eof"])
@pytest.mark.asyncio
async def test_established_reader_end_schedules_owner_cleanup(
    reader: object | None,
) -> None:
    process = _SlowReapProcess(reader)
    connection = _bare_connection(process)
    client = client_module.MCPClient(name="ended-reader-client")
    client.sessions["server"] = connection
    client.servers["server"] = {"command": "fake"}

    async def cleanup() -> None:
        await client._bounded_teardown_connection("server", session=connection)

    connection._on_transport_failure = cleanup
    connection._read_task = asyncio.create_task(connection._read_loop())
    try:
        await connection._read_task
        cleanup_task = connection._transport_cleanup_task
        assert cleanup_task is not None
        await asyncio.wait_for(process.wait_started.wait(), timeout=1)
        assert client.sessions == {"server": connection}

        process.allow_reap.set()
        await asyncio.wait_for(cleanup_task, timeout=1)

        assert process.stdin.closed
        assert process.returncode == 0
        assert connection._cleanup_complete is True
        assert connection._transport_cleanup_task is None
        assert client.sessions == {}
        assert client.servers == {}
    finally:
        process.allow_reap.set()
        cleanup_task = connection._transport_cleanup_task
        if cleanup_task is not None:
            await asyncio.gather(cleanup_task, return_exceptions=True)
        await connection.close()


@pytest.mark.asyncio
async def test_established_real_child_closing_stdout_is_reaped_without_disconnect() -> (
    None
):
    script = "import os,sys,time;sys.stdin.buffer.readline();os.close(1);time.sleep(60)"
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        limit=MAX_OUTPUT_LINE_BYTES,
    )
    client = client_module.MCPClient(name="real-eof-client")
    connection: client_module._StdioJSONRPCConnection

    async def cleanup() -> None:
        await client._bounded_teardown_connection("server", session=connection)

    connection = client_module._StdioJSONRPCConnection(
        process,
        client_name="real-eof-client",
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

        assert process.stdin.is_closing()
        assert process.returncode is not None
        assert connection._cleanup_complete is True
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


@pytest.mark.parametrize("owner_kind", ["active", "pending"])
@pytest.mark.parametrize("cancel_kind", ["direct", "wait_for"])
@pytest.mark.asyncio
async def test_cleanup_failure_replays_cancellation_and_retains_retryable_owner(
    owner_kind: str,
    cancel_kind: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = f"private-{owner_kind}-{cancel_kind}-payload"
    process = _RetryAfterPermissionErrorProcess(sentinel)
    close_started = asyncio.Event()
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            close_started.set()
            await asyncio.Future()

    session = Session()
    client = client_module.MCPClient(name="cancel-precedence-client")
    if owner_kind == "active":
        client.sessions["server"] = session  # type: ignore[assignment]
        client.servers["server"] = {"command": "fake"}
    else:
        client._pending_connections["server"] = SimpleNamespace(
            process=process, session=session
        )
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        client_module.logger,
        "warning",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    task = asyncio.create_task(client.disconnect_from_server("server"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    if cancel_kind == "direct":
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    else:
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(task, timeout=0.001)

    if owner_kind == "active":
        assert client.sessions == {"server": session}
        assert client.servers == {"server": {"command": "fake"}}
        assert client._pending_connections == {}
    else:
        assert client.sessions == {}
        assert client.servers == {}
        assert client._pending_connections["server"].session is session
    assert process.returncode is None
    assert sentinel not in repr(logged)
    assert any(
        args == ("MCP connection cleanup incomplete after cancellation",)
        for args, _kwargs in logged
    )

    assert await client.disconnect_from_server("server") is True
    assert process.returncode == -9
    assert process.kill_calls == 2
    assert client.sessions == {}
    assert client.servers == {}
    assert client._pending_connections == {}


@pytest.mark.asyncio
async def test_connect_cancellation_retains_pending_owner_when_reap_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sentinel = "private-connect-cancel-payload"
    process = _RetryAfterPermissionErrorProcess(sentinel)
    initialize_started = asyncio.Event()
    logged: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Session:
        def __init__(self, created_process: object, *, client_name: str, server_request_dispatcher: object | None = None) -> None:
            assert created_process is process
            self.process = created_process

        async def initialize(self) -> None:
            initialize_started.set()
            await asyncio.Future()

        async def close(self) -> None:
            await asyncio.Future()

    async def spawn(*_args: Any, **_kwargs: Any) -> _Process:
        return process

    client = client_module.MCPClient(name="connect-cancel-owner-client")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    monkeypatch.setattr(client_module, "_StdioJSONRPCConnection", Session)
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        client_module.logger,
        "warning",
        lambda *args, **kwargs: logged.append((args, kwargs)),
    )

    connect = asyncio.create_task(client.connect_to_server("server", "python"))
    await asyncio.wait_for(initialize_started.wait(), timeout=1)
    connect.cancel()
    with pytest.raises(asyncio.CancelledError):
        await connect

    assert client.sessions == {}
    assert client.servers == {}
    assert client._pending_connections["server"].session is not None
    assert process.returncode is None
    assert sentinel not in repr(logged)
    assert client._connect_reservations == {}

    assert await client.disconnect_from_server("server") is True
    assert process.returncode == -9
    assert client._pending_connections == {}


@pytest.mark.asyncio
async def test_disconnect_all_continues_after_cancelled_cleanup_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_started = asyncio.Event()
    first_process = _RetryAfterPermissionErrorProcess("private-all-cancel-payload")
    second_process = _Process()
    close_calls: list[str] = []

    class Session:
        def __init__(self, server_id: str, process: _Process) -> None:
            self.server_id = server_id
            self.process = process

        async def close(self) -> None:
            close_calls.append(self.server_id)
            if self.server_id == "first":
                first_started.set()
                await asyncio.Future()
            self.process.terminate()
            await self.process.wait()

    first_session = Session("first", first_process)
    second_session = Session("second", second_process)
    client = client_module.MCPClient(name="disconnect-all-failed-cleanup-client")
    client.sessions["first"] = first_session  # type: ignore[assignment]
    client.servers["first"] = {"command": "fake"}
    client._pending_connections["second"] = SimpleNamespace(
        process=second_process, session=second_session
    )
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    task = asyncio.create_task(client.disconnect_all())
    await asyncio.wait_for(first_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert close_calls == ["first", "second"]
    assert client.sessions == {"first": first_session}
    assert client.servers == {"first": {"command": "fake"}}
    assert client._pending_connections == {}
    assert first_process.returncode is None
    assert second_process.returncode == 0

    assert await client.disconnect_from_server("first") is True
    assert first_process.returncode == -9
    assert client.sessions == {}
    assert client.servers == {}


@pytest.mark.asyncio
async def test_disconnect_all_reports_incomplete_cleanup_and_continues(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first_process = _RetryAfterPermissionErrorProcess("private-all-false-payload")
    second_process = _Process()
    close_calls: list[str] = []
    warnings: list[tuple[tuple[object, ...], dict[str, object]]] = []
    infos: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Session:
        def __init__(self, server_id: str, process: _Process) -> None:
            self.server_id = server_id
            self.process = process

        async def close(self) -> None:
            close_calls.append(self.server_id)
            if self.server_id == "first":
                await asyncio.Future()
            self.process.terminate()
            await self.process.wait()

    first_session = Session("first", first_process)
    second_session = Session("second", second_process)
    client = client_module.MCPClient(name="disconnect-all-false-client")
    client.sessions["first"] = first_session  # type: ignore[assignment]
    client.servers["first"] = {"command": "fake"}
    client._pending_connections["second"] = SimpleNamespace(
        process=second_process, session=second_session
    )
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(
        client_module.logger,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )
    monkeypatch.setattr(
        client_module.logger,
        "info",
        lambda *args, **kwargs: infos.append((args, kwargs)),
    )

    await client.disconnect_all()

    assert close_calls == ["first", "second"]
    assert client.sessions == {"first": first_session}
    assert client.servers == {"first": {"command": "fake"}}
    assert client._pending_connections == {}
    assert first_process.returncode is None
    assert second_process.returncode == 0
    assert any(
        args == ("MCP disconnect_all cleanup incomplete",) for args, _kwargs in warnings
    )
    assert all(
        args != ("Disconnected from all MCP servers",) for args, _kwargs in infos
    )
    assert "private-all-false-payload" not in repr(warnings)

    assert await client.disconnect_from_server("first") is True
    assert first_process.returncode == -9
    assert client.sessions == {}
    assert client.servers == {}


@pytest.mark.asyncio
async def test_second_disconnect_cancellation_waits_for_retained_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    close_started = asyncio.Event()
    process = _SlowReapProcess()

    class Session:
        async def close(self) -> None:
            close_started.set()
            await asyncio.Future()

        def __init__(self) -> None:
            self.process = process

    session = Session()
    client = client_module.MCPClient(name="second-cancel-client")
    client.sessions["server"] = session  # type: ignore[assignment]
    client.servers["server"] = {"command": "fake"}
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    task = asyncio.create_task(client.disconnect_from_server("server"))
    await asyncio.wait_for(close_started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    waited_after_second_cancel = not task.done()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert waited_after_second_cancel
    assert process.stdin.closed
    assert process.returncode == -9
    assert process.kill_calls == 1
    assert client.sessions == {}
    assert client.servers == {}
    assert await client.disconnect_from_server("server") is False


@pytest.mark.asyncio
async def test_disconnect_wait_for_timeout_returns_only_after_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _SlowReapProcess()

    class Session:
        def __init__(self) -> None:
            self.process = process

        async def close(self) -> None:
            await asyncio.Future()

    client = client_module.MCPClient(name="wait-for-timeout-client")
    client.sessions["server"] = Session()  # type: ignore[assignment]
    client.servers["server"] = {"command": "fake"}
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(client.disconnect_from_server("server"), timeout=0.001)

    assert process.stdin.closed
    assert process.returncode == -9
    assert process.kill_calls == 1
    assert client.sessions == {}
    assert client.servers == {}


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
async def test_second_disconnect_all_cancellation_waits_for_every_cleanup(
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

    client = client_module.MCPClient(name="second-cancel-all-client")
    for server_id, process in zip(("first", "second"), processes):
        client.sessions[server_id] = Session(server_id, process)  # type: ignore[assignment]
        client.servers[server_id] = {"command": "fake"}
    monkeypatch.setattr(client_module, "CLEANUP_TIMEOUT_SECONDS", 0.01)
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.01)

    task = asyncio.create_task(client.disconnect_all())
    await asyncio.wait_for(first_started.wait(), timeout=1)
    task.cancel()
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.sleep(0)
    waited_after_second_cancel = not task.done()

    with pytest.raises(asyncio.CancelledError):
        await task
    first_returned_before_reap = processes[0].returncode is None
    await asyncio.wait_for(processes[0].allow_reap.wait(), timeout=1)
    await asyncio.sleep(0)

    assert waited_after_second_cancel
    assert not first_returned_before_reap
    assert close_calls == ["first", "second"]
    assert all(process.returncode is not None for process in processes)
    assert client.sessions == {}
    assert client.servers == {}
    await client.disconnect_all()


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
    ("list_method", "item_key", "item", "expected_name"),
    [
        pytest.param(
            "list_tools",
            "tools",
            {"name": "my tool", "inputSchema": {"type": "object"}},
            "my tool",
            id="tool-space",
        ),
        pytest.param(
            "list_tools",
            "tools",
            {"name": "", "inputSchema": {"type": "object"}},
            "",
            id="tool-empty",
        ),
        pytest.param(
            "list_prompts",
            "prompts",
            {"name": "my prompt", "arguments": []},
            "my prompt",
            id="prompt-space",
        ),
        pytest.param(
            "list_prompts",
            "prompts",
            {"name": "x" * 129, "arguments": []},
            "x" * 129,
            id="prompt-unbounded-by-later-profile",
        ),
    ],
)
@pytest.mark.asyncio
async def test_legacy_catalog_accepts_official_string_name_domain(
    list_method: str,
    item_key: str,
    item: dict[str, object],
    expected_name: str,
) -> None:
    connection, _requests = _scripted_connection(
        lambda _index, _method, _params: {item_key: [item]}
    )

    result = await getattr(connection, list_method)()

    assert getattr(result, item_key)[0].name == expected_name


@pytest.mark.parametrize(
    ("list_method", "item_key", "item"),
    [
        pytest.param(
            "list_tools",
            "tools",
            {"name": 7, "inputSchema": {"type": "object"}},
            id="tool-name-type",
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
            {"name": 7, "arguments": []},
            id="prompt-name-type",
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
