"""MCP hooks preserve the ordinary authorization and original result boundary."""

import asyncio
import json
from contextlib import asynccontextmanager
from dataclasses import replace

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.Agents.test_mcp_tool_provider import (
    FakeMCPService,
    _catalog_record,
    _tool_dict,
)
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
from tldw_chatbook.MCP.permission_store import EffectiveToolState
from tldw_chatbook.MCP.tool_results import decode_protocol_frame, parse_tool_result


def wire_result(decision="pass"):
    frame = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "structuredContent": {"version": 2, "decision": decision},
                "content": [],
            },
        }
    ).encode()
    return parse_tool_result(decode_protocol_frame(frame)["result"])


@pytest.mark.asyncio
async def test_capture_requires_exact_normal_result_and_clears_after_scope():
    from tldw_chatbook.Agents.mcp_tool_provider import capture_mcp_result

    raw = wire_result()
    service = FakeMCPService(
        catalog_records=[
            _catalog_record(
                "test", [_tool_dict("check", input_schema={"type": "object"})]
            )
        ],
        default_state=EffectiveToolState(state="allow", origin="global_default"),
        execute_result=raw,
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.get_running_loop())
    await provider.compose_catalog()
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    name = provider.list_catalog()[0].name

    def invoke():
        with capture_mcp_result(provider) as capture:
            result = registry.invoke_by_name(
                name, {}, expected_definition=registry.snapshot_for_hook(name)
            )
            assert result.ok
            assert (
                capture.consume(
                    replace(result, ok=False, error="plugin_result_revoked")
                )
                is None
            )
        with capture_mcp_result(provider) as capture:
            result = registry.invoke_by_name(name, {})
            assert capture.consume(result) is raw
            assert capture.consume(result) is None
        assert capture.consume(result) is None
        service.kill_switch = True
        with capture_mcp_result(provider) as capture:
            refused = registry.invoke_by_name(name, {})
            assert not refused.ok
            assert capture.consume(refused) is None
        assert len(service.execute_calls) == 2

    await asyncio.to_thread(invoke)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "verdict",
    [
        "approve_once",
        "approve_session",
        "always_allow",
        "allow_matching",
        "cached_session",
        "unanswered",
    ],
)
async def test_normal_approval_metadata_preserves_exact_result_capture(verdict):
    from tldw_chatbook.Agents.mcp_tool_provider import capture_mcp_result

    raw = wire_result()
    service = FakeMCPService(
        catalog_records=[_catalog_record("test", [_tool_dict("check")])],
        execute_result=raw,
    )

    class Decisions(dict):
        unresolved_keys = frozenset()

    def approve(calls):
        decisions = Decisions(
            {
                call.llm_name: "approve_once" if verdict == "unanswered" else verdict
                for call in calls
            }
        )
        if verdict == "unanswered":
            decisions.unresolved_keys = frozenset(decisions)
        return decisions

    provider = MCPToolProvider(
        service=service, main_loop=asyncio.get_running_loop(), approval_callback=approve
    )
    await provider.compose_catalog()
    name = provider.list_catalog()[0].name
    if verdict == "cached_session":
        tool, _state = provider._entry_by_llm_name[name]
        service.approve_for_session(tool.server_key, tool.name)

    def invoke():
        with capture_mcp_result(provider) as capture:
            result = provider.invoke(name, {})
            assert result.ok and len(service.execute_calls) == 1
            assert result.approval_decision == (
                None if verdict == "unanswered" else "approved"
            )
            assert capture.consume(result) is raw
            assert capture.consume(result) is None

    await asyncio.to_thread(invoke)


@pytest.mark.asyncio
async def test_capture_nested_and_concurrent_requests_are_isolated():
    from tldw_chatbook.Agents.mcp_tool_provider import capture_mcp_result

    raw = wire_result()
    service = FakeMCPService(
        catalog_records=[_catalog_record("test", [_tool_dict("check")])],
        default_state=EffectiveToolState(state="allow", origin="global_default"),
        execute_result=raw,
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.get_running_loop())
    await provider.compose_catalog()
    name = provider.list_catalog()[0].name

    def invoke():
        with capture_mcp_result(provider) as outer:
            with capture_mcp_result(provider) as inner:
                result = provider.invoke(name, {})
                assert inner.consume(result) is raw
            assert outer.consume(result) is None
        return provider.invoke(name, {}).ok

    assert all(await asyncio.gather(*(asyncio.to_thread(invoke) for _ in range(3))))


def test_typed_templates_expand_once_and_preserve_types():
    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import expand_input
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    handler = parse_handlers(
        [
            {
                "id": "template",
                "event": "PreToolUse",
                "type": "mcp_tool",
                "server": "local:test",
                "tool": "check",
                "effects": [],
                "input": {
                    "value": "${data.tool_args}",
                    "label": "tool:${data.tool_name}",
                },
            }
        ]
    )[0]
    value = {
        key: item for key, item in event().model_dump().items() if item is not None
    }
    value["data"].update(
        tool_args={"literal": "${run_id}", "boolean": True}, tool_name="scanner"
    )
    from tldw_chatbook.Agents.hooks_v2.validation import parse_event

    expanded = expand_input(parse_event(value), handler)
    assert expanded == {
        "value": {"literal": "${run_id}", "boolean": True},
        "label": "tool:scanner",
    }
    with pytest.raises(ValueError):
        expand_input(event(), handler)


def test_host_causal_chain_refuses_cycles_and_depth_five():
    from tldw_chatbook.Agents.hooks_v2.causality import CausalChain, current_chain

    chain = CausalChain().enter("event-a", "handler-a", "tool-a")
    with chain.scope():
        assert current_chain() is chain
    assert current_chain().visits == ()
    with pytest.raises(ValueError, match="cycle"):
        chain.enter("different-event", "handler-a", "tool-b")
    with pytest.raises(ValueError, match="cycle"):
        chain.enter("different-event", "handler-b", "tool-a")
    for i in range(1, 4):
        chain = chain.enter(f"event-{i}", f"handler-{i}", f"tool-{i}")
    with pytest.raises(ValueError, match="depth"):
        chain.enter("event-5", "handler-5", "tool-5")


@pytest.mark.asyncio
async def test_cancelled_reacquire_keeps_lifetime_until_actual_owner_release():
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner

    owner = HookBudgetOwner()
    ticket = owner.reserve("a", False)
    await ticket.acquire()
    ticket.suspend()
    occupied = [owner.reserve("a", False) for _ in range(4)]
    await asyncio.gather(*(item.acquire() for item in occupied))
    waiting = asyncio.create_task(ticket.acquire(retain_on_cancel=True))
    await asyncio.sleep(0)
    waiting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await waiting
    assert not ticket.released
    assert owner.snapshot("a") == {
        "execution": 4,
        "tickets": 5,
        "observations": 0,
        "workers": 0,
    }
    for item in occupied:
        item.release()
    await ticket.acquire(retain_on_cancel=True)
    ticket.release()
    assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("is_error", [False, True])
async def test_real_stdio_normal_registry_preserves_original_hook_payload(
    tmp_path, is_error
):
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.mcp_results import normalize_hook_result
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
    from tldw_chatbook.Agents.mcp_tool_provider import capture_mcp_result
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    handler = parse_handlers(
        [
            {
                "id": "stdio-hook",
                "event": "PreToolUse",
                "type": "mcp_tool",
                "server": "local:fixture",
                "tool": "fixture",
                "effects": ["deny"],
            }
        ]
    )[0]
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        provider = MCPToolProvider(
            service=service, main_loop=asyncio.get_running_loop()
        )
        await provider.compose_catalog()
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        service.set_tool_state("local:fixture", "fixture", "allow", tool=tool)
        registry = ToolCatalogRegistry()
        registry.register_provider(provider)
        name = provider.list_catalog()[0].name
        control = await asyncio.to_thread(
            registry.invoke_by_name,
            name,
            {"payload": {"content": [{"type": "text", "text": "ordinary"}]}},
        )
        assert control.ok and "ordinary" in control.content

        def invoke():
            with capture_mcp_result(provider) as capture:
                result = registry.invoke_by_name(
                    name,
                    {
                        "payload": {
                            "isError": is_error,
                            "structuredContent": {"version": 2, "decision": "pass"},
                            "content": [],
                            "_meta": {"trace": "retained"},
                        }
                    },
                    expected_definition=registry.snapshot_for_hook(name),
                )
                raw = capture.consume(result)
                assert raw.duplicate_keys_checked
                assert raw.metadata == {"trace": "retained"}
                assert result.dispatch_state == "settled"
                if is_error:
                    assert not result.ok
                    with pytest.raises(ValueError, match="execution_failed"):
                        normalize_hook_result(raw, handler)
                else:
                    assert result.ok
                    assert normalize_hook_result(raw, handler).decision == "pass"

        await asyncio.to_thread(invoke)
        assert len(service.execution_log.read_recent()) == 2


@pytest.mark.asyncio
async def test_restricted_normal_provider_cannot_prompt_or_use_old_stamp():
    import contextlib
    import threading
    import time

    from tldw_chatbook.Agents.mcp_tool_provider import (
        MCPInvocationPolicy,
        restrict_mcp_invocation,
    )
    from tldw_chatbook.Agents.run_context import use_run_id

    prompted = []
    service = FakeMCPService(
        catalog_records=[_catalog_record("test", [_tool_dict("check")])],
        execute_result=wire_result(),
    )
    provider = MCPToolProvider(
        service=service,
        main_loop=asyncio.get_running_loop(),
        approval_callback=lambda calls: prompted.append(calls) or {},
    )
    await provider.compose_catalog()
    name = provider.list_catalog()[0].name
    provider.apply_batch_decisions("parent", {name: "approve_once"})
    policy = MCPInvocationPolicy(
        current=lambda: True,
        deadline=time.monotonic() + 10,
        cancel_event=threading.Event(),
        allow_approval=False,
        wait_scope=lambda _kind: contextlib.nullcontext(),
    )

    def invoke():
        with use_run_id("parent"), restrict_mcp_invocation(policy):
            return provider.invoke(name, {})

    result = await asyncio.to_thread(invoke)
    assert not result.ok and not service.execute_calls and not prompted


@pytest.mark.asyncio
async def test_restricted_actual_local_boundary_refuses_disconnected_reconnect(
    tmp_path,
):
    import contextlib
    import threading
    import time

    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.mcp_tool_provider import (
        MCPInvocationPolicy,
        restrict_mcp_invocation,
    )

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        policy = MCPInvocationPolicy(
            current=lambda: True,
            deadline=time.monotonic() + 10,
            cancel_event=threading.Event(),
            allow_approval=False,
            wait_scope=lambda _kind: contextlib.nullcontext(),
        )
        with restrict_mcp_invocation(policy):
            result = await service.execute_hub_tool_result(
                "local:fixture", "fixture", {}
            )
            assert result.transport_error is None
        await client.disconnect_from_server("fixture")
        with restrict_mcp_invocation(policy), pytest.raises(PermissionError):
            await service.execute_hub_tool_result("local:fixture", "fixture", {})
        assert not client.sessions


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "cyclic_guard",
        "depends_on_initializer",
        "unknown_dependencies",
        "disconnected",
        "cyclic_post",
    ],
)
async def test_actual_stdio_required_initializer_runs_through_existing_tool_pipeline(
    tmp_path, mode
):
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    body = {
        "version": 2,
        "decision": "pass",
        "context": [{"text": "real MCP initialization", "lifetime": "runtime"}],
    }
    handler = parse_handlers(
        [
            {
                "id": "initializer",
                "event": "SessionStart",
                "type": "mcp_tool",
                "server": "local:fixture",
                "tool": "fixture",
                "required": True,
                "effects": ["context"],
                "require_context": True,
                "input": {"payload": {"structuredContent": body}},
            }
        ]
    )[0]
    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        provider = MCPToolProvider(
            service=service,
            main_loop=asyncio.get_running_loop(),
            approval_callback=lambda calls: {
                call.llm_name: "approve_once" for call in calls
            },
        )
        await provider.compose_catalog()
        registry = ToolCatalogRegistry()
        registry.register_provider(provider)
        owner = HookBudgetOwner()
        definitions = [handler]
        if mode in {"cyclic_guard", "cyclic_post"}:
            definitions += parse_handlers(
                [
                    {
                        "id": "guard",
                        "event": (
                            "PreToolUse" if mode == "cyclic_guard" else "PostToolUse"
                        ),
                        "type": "mcp_tool",
                        "server": "local:fixture",
                        "tool": "fixture",
                        "required": True,
                        "effects": ["deny"] if mode == "cyclic_guard" else [],
                        "input": {},
                    }
                ]
            )
        engine = HookEngine(tuple(definitions), lambda *_: True, owner)
        lifecycle = HookSessionLifecycle(engine, "session")
        context = MCPHookContext(
            registry=registry,
            lifecycle=lifecycle,
            parent_scope=lifecycle.scope_id,
            run_id="reserved-provisional",
            session_id="session",
            allowed_names=frozenset(entry.name for entry in registry.list_catalog()),
            current=lambda: True,
            required_handler_ids=lambda _definition: (
                None
                if mode == "unknown_dependencies"
                else ("initializer",)
                if mode == "depends_on_initializer"
                else ()
            ),
        )
        engine.mcp_executor.bind_context(context)
        if mode == "disconnected":
            await client.disconnect_from_server("fixture")
        token = lifecycle.reserve(
            lifecycle.event("SessionStart", data={"reason": "startup"})
        )
        try:
            result = await asyncio.wait_for(lifecycle.initialize(token), 5)
            if mode != "success":
                assert not result.allowed and not result.accepted
                if mode == "disconnected":
                    assert not client.sessions
                    assert len(service.execution_log.read_recent()) == 1
                    assert (
                        service.execution_log.read_recent()[0]["error_category"]
                        == "definition_changed"
                    )
                elif mode == "cyclic_post":
                    assert len(service.execution_log.read_recent()) == 1
                else:
                    assert not service.execution_log.read_recent()
                assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
                return
            assert result.succeeded, result
            lifecycle.publish(token)
            assert lifecycle.live
            assert "real MCP initialization" in str(
                lifecycle.context.blocks(lifecycle.scope_id, "model")
            )
            assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
            assert len(service.execution_log.read_recent()) == 1
        finally:
            await engine.close()


@pytest.mark.asyncio
async def test_provider_lock_wait_suspends_and_cancel_before_dispatch_releases_waiter():
    import contextlib
    import threading
    import time

    from tldw_chatbook.Agents.mcp_tool_provider import (
        MCPInvocationPolicy,
        restrict_mcp_invocation,
    )

    service = FakeMCPService(
        catalog_records=[_catalog_record("test", [_tool_dict("check")])],
        default_state=EffectiveToolState(state="allow", origin="global_default"),
        execute_result=wire_result(),
    )
    provider = MCPToolProvider(service=service, main_loop=asyncio.get_running_loop())
    await provider.compose_catalog()
    name = provider.list_catalog()[0].name
    waiting, finished = threading.Event(), threading.Event()

    @contextlib.contextmanager
    def wait_scope(kind):
        assert kind == "provider"
        waiting.set()
        try:
            yield
        finally:
            finished.set()

    policy = MCPInvocationPolicy(
        current=lambda: True,
        deadline=time.monotonic() + 10,
        cancel_event=threading.Event(),
        allow_approval=False,
        wait_scope=wait_scope,
    )

    def invoke():
        with restrict_mcp_invocation(policy):
            return provider.invoke(name, {})

    provider._invoke_lock.acquire()
    pending = asyncio.create_task(asyncio.to_thread(invoke))
    try:
        for _ in range(30):
            if waiting.is_set():
                break
            await asyncio.sleep(0.01)
        assert waiting.is_set()
        policy.cancel_event.set()
        result = await asyncio.wait_for(pending, 1)
        assert not result.ok and result.dispatch_state == "not_started"
        assert finished.is_set() and not service.execute_calls
    finally:
        provider._invoke_lock.release()
        await asyncio.wait_for(pending, 1)
    # The cancelled waiter never acquired or poisoned the ordinary lock.
    assert (await asyncio.to_thread(provider.invoke, name, {})).ok


@asynccontextmanager
async def http_hook_service(tmp_path, monkeypatch):
    """An actual HTTP peer, allowing only its exact owned loopback address."""
    import _socket
    import socket

    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import (
        LocalExternalMCPProfile,
        LocalMCPStore,
        TransportProfile,
    )
    from tldw_chatbook.MCP.unified_control_plane_service import (
        UnifiedMCPControlPlaneService,
    )

    messages, writers, tasks = [], set(), set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        writers.add(writer)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("ascii").split("\r\n")
            verb = lines[0].split()[0]
            headers = dict(
                line.lower().split(": ", 1) for line in lines[1:] if ": " in line
            )
            body = await reader.readexactly(int(headers.get("content-length", "0")))
            message = json.loads(body) if body else {}
            messages.append(message)
            extra = ""
            if verb == "DELETE" or "id" not in message:
                status, payload = 202, b""
            else:
                method = message["method"]
                if method in {"initialize", "server/discover"}:
                    result = {
                        "protocolVersion": "2025-03-26",
                        "supportedVersions": ["2025-03-26"],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "hooks", "version": "1"},
                    }
                    extra = "Mcp-Session-Id: hook-owned\r\n"
                elif method == "tools/list":
                    result = {
                        "tools": [
                            {
                                "name": "fixture",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {"payload": {"type": "object"}},
                                },
                            }
                        ]
                    }
                elif method == "tools/call":
                    result = message["params"]["arguments"]["payload"]
                else:
                    raise AssertionError(method)
                status = 200
                payload = json.dumps(
                    {"jsonrpc": "2.0", "id": message["id"], "result": result}
                ).encode()
            writer.write(
                f"HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\n{extra}Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            writers.discard(writer)
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    old_connect, old_dns = socket.socket.connect, socket.getaddrinfo

    def connect(sock, address):
        if sock.family == socket.AF_INET and address == ("127.0.0.1", port):
            return _socket.socket.connect(sock, address)
        return old_connect(sock, address)

    def dns(host, service, *args, **kwargs):
        if host in ("127.0.0.1", b"127.0.0.1") and int(service) == port:
            return _socket.getaddrinfo(host, service, *args, **kwargs)
        return old_dns(host, service, *args, **kwargs)

    client = MCPClient()
    with monkeypatch.context() as scoped:
        scoped.setattr(socket.socket, "connect", connect)
        scoped.setattr(socket, "getaddrinfo", dns)
        try:
            profile = TransportProfile(
                profile_id="fixture",
                transport="streamable_http",
                url=f"http://127.0.0.1:{port}/mcp",
                protocol_version="2025-03-26",
                development_loopback=True,
            )
            assert await client.connect_profile(profile)
            store = LocalMCPStore(tmp_path / "http-store.json")
            store.save_profile(
                LocalExternalMCPProfile(
                    profile_id="fixture",
                    transport="streamable_http",
                    url=profile.url,
                    protocol_version=profile.protocol_version,
                    development_loopback=True,
                )
            )
            store.save_discovery_snapshot(
                "fixture", await client.describe_server("fixture")
            )
            service = UnifiedMCPControlPlaneService(
                local_service=LocalMCPControlService(
                    store=store, client=client, manifest_provider=dict
                ),
                server_service=None,
                target_store=None,
                context_store=None,
            )
            yield service, client, messages
        finally:
            await client.disconnect_from_server("fixture")
            assert not client.sessions
            server.close()
            await server.wait_closed()
            for writer in tuple(writers):
                writer.close()
            pending = tuple(tasks)
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["stdio", "http"])
@pytest.mark.parametrize(
    "payload,success",
    [
        ({"structuredContent": {"version": 2, "decision": "pass"}}, True),
        (
            {"content": [{"type": "text", "text": '{"version":2,"decision":"pass"}'}]},
            True,
        ),
        (
            {"structuredContent": {"version": 2, "decision": "pass"}, "isError": True},
            False,
        ),
        (
            {
                "structuredContent": {"version": 2, "decision": "pass"},
                "_meta": {"padding": "x" * 16384},
            },
            False,
        ),
        (
            {
                "structuredContent": {"version": 2, "decision": "pass"},
                "content": [
                    {"type": "text", "text": '{"version":2,"decision":"deny"}'}
                ],
            },
            False,
        ),
    ],
)
async def test_actual_transport_engine_result_boundary(
    tmp_path, monkeypatch, transport, payload, success
):
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    @asynccontextmanager
    async def fixture():
        if transport == "http":
            async with http_hook_service(tmp_path, monkeypatch) as (
                service,
                client,
                _messages,
            ):
                yield service, client
        else:
            async with controlled_stdio_client(tmp_path) as client:
                yield await controlled_service(tmp_path, client), client

    async with fixture() as (service, client):
        provider = MCPToolProvider(
            service=service,
            main_loop=asyncio.get_running_loop(),
            approval_callback=lambda calls: {
                call.llm_name: "approve_once" for call in calls
            },
        )
        await provider.compose_catalog()
        registry = ToolCatalogRegistry()
        registry.register_provider(provider)
        names = frozenset(entry.name for entry in registry.list_catalog())
        assert len(names) == 1 and client.sessions
        handler = parse_handlers(
            [
                {
                    "id": "initial",
                    "event": "SessionStart",
                    "type": "mcp_tool",
                    "server": "local:fixture",
                    "tool": "fixture",
                    "required": True,
                    "effects": [],
                    "input": {"payload": payload},
                }
            ]
        )[0]
        owner = HookBudgetOwner()
        engine = HookEngine((handler,), lambda *_: True, owner)
        lifecycle = HookSessionLifecycle(engine, "session")
        engine.mcp_executor.bind_context(
            MCPHookContext(
                registry,
                lifecycle,
                lifecycle.scope_id,
                "reserved",
                "session",
                names,
                lambda: True,
                lambda _definition: (),
            )
        )
        token = lifecycle.reserve(
            lifecycle.event("SessionStart", data={"reason": "startup"})
        )
        try:
            outcome = await asyncio.wait_for(lifecycle.initialize(token), 5)
            assert outcome.succeeded is success, outcome
            assert len(service.execution_log.read_recent()) == 1
            assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
        finally:
            lifecycle.cancel(token)
            await engine.close()


def test_nested_context_waits_for_containing_acceptance_and_exact_input_owner():
    from Tests.Agents.test_hooks_v2_execution import event
    from tldw_chatbook.Agents.hooks_v2.checkpoints import HookCheckpointStore
    from tldw_chatbook.Agents.hooks_v2.context import ContextLedger
    from tldw_chatbook.Agents.hooks_v2.engine import HookEventOutcome
    from tldw_chatbook.Agents.hooks_v2.models import HookResult
    from tldw_chatbook.Agents.hooks_v2.validation import parse_event

    store = HookCheckpointStore()
    store.bind_owner("runtime")
    store.bind_owner("pending-input", "runtime")
    ledger = ContextLedger(owners=store.owners)
    from tldw_chatbook.Agents.hooks_v2.lifecycle import lifecycle_event

    parent = lifecycle_event("SessionStart", "session", data={"reason": "startup"})
    child_data = event().model_dump()
    child_data["event_id"] = "nested-event"
    child = parse_event(
        {key: value for key, value in child_data.items() if value is not None}
    )
    outcome = HookEventOutcome(
        accepted=(("nested-context", HookResult(version=2, decision="pass")),)
    )
    current = [True]
    ledger.stage_nested(
        parent,
        "initializer",
        "pending-input",
        child,
        outcome,
        ({"role": "user", "content": "nested turn instructions"},),
        current=lambda: current[0],
    )
    assert not ledger.blocks("pending-input", "before")
    ledger.bind("runtime", parent)
    ledger.accept(
        parent,
        HookEventOutcome(
            accepted=(("initializer", HookResult(version=2, decision="pass")),)
        ),
    )
    assert "nested turn instructions" in str(ledger.blocks("pending-input", "model"))
    store.bind_owner("second-input", "runtime")
    assert not ledger.blocks("second-input", "model")
    current[0] = False
    with pytest.raises(ValueError, match="stale"):
        ledger.blocks("pending-input", "another-boundary")


@pytest.mark.asyncio
async def test_admitted_agent_service_uses_same_mcp_hook_executor_and_parent_context(
    tmp_path,
):
    from Tests.Agents.test_agent_service import CFG, ScriptedChat, fence
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
    from tldw_chatbook.Agents.tool_catalog import BuiltinToolProvider
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    async with controlled_stdio_client(tmp_path) as client:
        normal_service = await controlled_service(tmp_path, client)
        provider = MCPToolProvider(
            service=normal_service,
            main_loop=asyncio.get_running_loop(),
            approval_callback=lambda calls: {
                call.llm_name: "approve_once" for call in calls
            },
        )
        await provider.compose_catalog()
        name = provider.list_catalog()[0].name
        registry = ToolCatalogRegistry()
        registry.register_provider(BuiltinToolProvider())
        registry.register_provider(provider)
        handlers = parse_handlers(
            [
                {
                    "id": "review-calculator",
                    "event": "PreToolUse",
                    "type": "mcp_tool",
                    "match": {
                        "tool_id": [registry.snapshot_for_hook("calculator").tool_id]
                    },
                    "server": "local:fixture",
                    "tool": "fixture",
                    "required": True,
                    "effects": ["context"],
                    "require_context": True,
                    "input": {
                        "payload": {
                            "structuredContent": {
                                "version": 2,
                                "decision": "pass",
                                "context": [
                                    {
                                        "text": "Admitted MCP review context",
                                        "lifetime": "turn",
                                    }
                                ],
                            }
                        }
                    },
                }
            ]
        )
        owner = HookBudgetOwner()
        engine = HookEngine(handlers, lambda *_: True, owner)
        db = AgentRunsDB(tmp_path / "agent-runs.db", client_id="h6-actual")
        chat = ScriptedChat([fence("calculator", {"expression": "6*7"}), "done"])
        service = AgentService(
            db,
            registry,
            chat_call=chat,
            hooks_v2_engine=engine,
            hooks_v2_session_id="session",
            hooks_v2_turn_id="turn",
        )
        try:
            _run_id, result = await asyncio.wait_for(
                asyncio.to_thread(
                    service.run_turn,
                    conversation_id="conversation",
                    messages=[{"role": "user", "content": "calculate"}],
                    config=replace(CFG, allowed_tools=("calculator", name)),
                    api_endpoint="test",
                ),
                10,
            )
            assert result.status == "done", result
            assert len(normal_service.execution_log.read_recent()) == 1
            assert "Admitted MCP review context" in str(chat.calls[-1])
            assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
        finally:
            await engine.close()
            db.close()


@pytest.mark.asyncio
async def test_mcp_executor_installs_both_required_post_events_before_either_settles(
    tmp_path,
):
    from Tests.Agents.test_hooks_v2_tool_pipeline import hook_command
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers

    first, release, second = (
        tmp_path / "post1",
        tmp_path / "release1",
        tmp_path / "post2",
    )
    bodies = []
    for marker, code in (
        (first, f"while not Path({str(release)!r}).exists(): time.sleep(.01)"),
        (second, "raise SystemExit(1)"),
    ):
        bodies.append(
            "from pathlib import Path;import time;"
            + f"Path({str(marker)!r}).touch();exec({code!r})"
        )
    initializer = parse_handlers(
        [
            {
                "id": "initializer",
                "event": "SessionStart",
                "type": "mcp_tool",
                "server": "local:fixture",
                "tool": "fixture",
                "required": True,
                "effects": [],
                "input": {
                    "payload": {
                        "structuredContent": {"version": 2, "decision": "pass"},
                        "isError": True,
                    }
                },
            }
        ]
    )[0]
    handlers = (
        initializer,
        hook_command("post1", event="PostToolUse", code=bodies[0], required=True),
        hook_command(
            "post2", event="PostToolUseFailure", code=bodies[1], required=True
        ),
    )
    operations = []

    def authority(handler, event, stage):
        if handler.id == "post1" and stage == "launch":
            operations.append(event.run_id)
        return True

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        provider = MCPToolProvider(
            service=service,
            main_loop=asyncio.get_running_loop(),
            approval_callback=lambda calls: {
                call.llm_name: "approve_once" for call in calls
            },
        )
        await provider.compose_catalog()
        registry = ToolCatalogRegistry()
        registry.register_provider(provider)
        owner = HookBudgetOwner()
        engine = HookEngine(handlers, authority, owner)
        lifecycle = HookSessionLifecycle(engine, "session")
        engine.mcp_executor.bind_context(
            MCPHookContext(
                registry,
                lifecycle,
                lifecycle.scope_id,
                "reserved",
                "session",
                frozenset(row.name for row in registry.list_catalog()),
                lambda: True,
                lambda _definition: (),
            )
        )
        pending = asyncio.create_task(
            engine.fire_async(
                lifecycle.event("SessionStart", data={"reason": "startup"})
            )
        )
        try:
            for _ in range(400):
                if first.exists():
                    break
                await asyncio.sleep(0.01)
            assert first.exists() and not second.exists() and not pending.done()
            assert operations
            with pytest.raises(RuntimeError, match="pending"):
                lifecycle.checkpoints.assert_continuation_dependencies(
                    operations[0], required_handler_ids=("post2",)
                )
            assert (
                owner.snapshot()["tickets"] == 2 and owner.snapshot()["execution"] == 1
            )
            release.touch()
            outcome = await asyncio.wait_for(pending, 5)
            assert second.exists() and not outcome.allowed and not outcome.accepted
            assert len(service.execution_log.read_recent()) == 1
            assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
        finally:
            release.touch()
            await asyncio.wait_for(pending, 5)
            await engine.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "approval,change,phase",
    [
        ("allow", "none", "post"),
        ("allow", "deny", "post"),
        ("once", "none", "post"),
        ("once", "deny", "post"),
        ("allow", "ask", "post"),
        ("once", "profile", "post"),
        ("once", "definition", "post"),
        ("once", "discovery", "post"),
        ("once", "discovery", "accept"),
        ("once", "kill_unavailable", "post"),
        ("once", "persona_unavailable", "post"),
        ("allow", "deny", "accept"),
        ("once", "none", "accept"),
    ],
)
async def test_original_mcp_result_rechecks_normal_authority_after_post_and_at_acceptance(
    tmp_path, approval, change, phase
):
    import threading

    from Tests.Agents.test_hooks_v2_tool_pipeline import hook_command
    from Tests.MCP.test_typed_tool_results import (
        controlled_service,
        controlled_stdio_client,
    )
    from tldw_chatbook.Agents.hooks_v2.budgets import HookBudgetOwner
    from tldw_chatbook.Agents.hooks_v2.engine import HookEngine
    from tldw_chatbook.Agents.hooks_v2.lifecycle import HookSessionLifecycle
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookContext
    from tldw_chatbook.Agents.hooks_v2.validation import parse_handlers
    from tldw_chatbook.MCP.hub_tool_catalog import HubTool

    marker, release = tmp_path / "post-entered", tmp_path / "post-release"
    accept_entered, accept_release = threading.Event(), threading.Event()
    profile = ["default"]
    prompts = []
    initializer = parse_handlers(
        [
            {
                "id": "init",
                "event": "SessionStart",
                "type": "mcp_tool",
                "server": "local:fixture",
                "tool": "fixture",
                "required": True,
                "effects": ["context"],
                "require_context": True,
                "input": {
                    "payload": {
                        "structuredContent": {
                            "version": 2,
                            "decision": "pass",
                            "context": [
                                {
                                    "text": "revocable instructions",
                                    "lifetime": "runtime",
                                }
                            ],
                        }
                    }
                },
            }
        ]
    )[0]
    wait = f"while not Path({str(release)!r}).exists(): time.sleep(.01)"
    post = hook_command(
        "post",
        event="PostToolUse",
        required=True,
        code=f"from pathlib import Path;import time;Path({str(marker)!r}).touch();exec({wait!r})",
    )

    def authority(handler, _event, stage):
        if phase == "accept" and handler.id == "init" and stage == "accept":
            accept_entered.set()
            return accept_release.wait(5)
        return True

    async with controlled_stdio_client(tmp_path) as client:
        service = await controlled_service(tmp_path, client)
        tool = HubTool(
            server_key="local:fixture",
            server_label="fixture",
            source="local",
            name="fixture",
            description="",
            input_schema={"type": "object"},
            tags=(),
            stale=False,
            executable=True,
        )
        if approval == "allow":
            service.set_tool_state(tool.server_key, tool.name, "allow", tool=tool)

        def approve(calls):
            prompts.extend(calls)
            return {call.llm_name: "approve_once" for call in calls}

        provider = MCPToolProvider(
            service=service,
            main_loop=asyncio.get_running_loop(),
            approval_callback=approve,
            profile_id_provider=lambda: profile[0],
        )
        await provider.compose_catalog()
        registry = ToolCatalogRegistry()
        registry.register_provider(provider)
        name = provider.list_catalog()[0].name
        owner = HookBudgetOwner()
        engine = HookEngine((initializer, post), authority, owner)
        lifecycle = HookSessionLifecycle(engine, "session")
        engine.mcp_executor.bind_context(
            MCPHookContext(
                registry,
                lifecycle,
                lifecycle.scope_id,
                "reserved",
                "session",
                frozenset(row.name for row in registry.list_catalog()),
                lambda: True,
                lambda _definition: (),
            )
        )
        pending = asyncio.create_task(
            engine.fire_async(
                lifecycle.event("SessionStart", data={"reason": "startup"})
            )
        )
        try:
            for _ in range(300):
                if marker.exists():
                    break
                await asyncio.sleep(0.01)
            assert marker.exists() and not pending.done()
            assert len(service.execution_log.read_recent()) == 1
            assert len(prompts) == (1 if approval == "once" else 0)
            if phase == "accept":
                release.touch()
                assert await asyncio.to_thread(accept_entered.wait, 5)
                assert not pending.done()
            if change in {"deny", "ask"}:
                service.set_tool_state(tool.server_key, tool.name, change, tool=tool)
                assert service.gate_tool_test(tool).state == change
                if change == "deny":
                    control = await asyncio.to_thread(provider.invoke, name, {})
                    assert not control.ok and control.dispatch_state == "not_started"
            elif change == "profile":
                profile[0] = "changed-profile"
            elif change in {"kill_unavailable", "persona_unavailable"}:

                def unavailable():
                    raise RuntimeError("current configured authority unavailable")

                if change == "kill_unavailable":
                    service.get_kill_switch = unavailable
                else:
                    provider._persona_policy_provider = unavailable
            elif change == "discovery":
                discovery = await client.describe_server("fixture")
                discovery["tools"][0]["description"] = "changed live definition"
                service.local_service.store.save_discovery_snapshot(
                    "fixture", discovery
                )
            elif change == "definition":
                original, state = provider._entry_by_llm_name[name]
                provider._entry_by_llm_name[name] = (
                    replace(
                        original,
                        input_schema={"type": "object", "required": ["changed"]},
                    ),
                    state,
                )
            release.touch()
            accept_release.set()
            outcome = await asyncio.wait_for(pending, 5)
            assert bool(outcome.accepted) is (change == "none"), outcome
            assert outcome.allowed is (change == "none")
            assert len(prompts) == (1 if approval == "once" else 0)
            assert owner.snapshot()["tickets"] == owner.snapshot()["execution"] == 0
            counter = await client.call_tool_result(
                "fixture", "fixture", {"counter": True}
            )
            assert counter.content[0]["text"] == "2", (
                "acceptance replayed the original tool"
            )
        finally:
            release.touch()
            accept_release.set()
            if not pending.done():
                pending.cancel()
            await asyncio.gather(pending, return_exceptions=True)
            await engine.close()
