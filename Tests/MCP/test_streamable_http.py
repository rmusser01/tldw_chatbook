"""Direct MCP wire qualification against owned loopback peers."""


def test_unknown_protocol_does_not_fall_back_to_old_handshake():
    import pytest

    from tldw_chatbook.MCP.protocol_profiles import protocol_profile

    with pytest.raises(ValueError):
        protocol_profile("2099-01-01")
    assert protocol_profile("2025-03-26").version == "2025-03-26"


import _socket
import asyncio
import json
import socket
from contextlib import asynccontextmanager

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

VERSIONS = ("2026-07-28", "2025-11-25", "2025-03-26")


@asynccontextmanager
async def peer(
    monkeypatch, *, version="2026-07-28", mode="json", fault=None, supported=None
):
    """One owned address/port; preserve the caller's refusal everywhere else."""
    exchanges, writers, tasks = [], set(), set()
    calls = {"count": 0, "targets": []}
    resume = {}

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        writers.add(writer)
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("ascii").split("\r\n")
            verb, target, _http_version = lines[0].split()
            calls["targets"].append((verb, target))
            headers = dict(line.split(": ", 1) for line in lines[1:] if ": " in line)
            headers = {key.lower(): value for key, value in headers.items()}
            body = await reader.readexactly(int(headers.get("content-length", "0")))
            message = json.loads(body) if body else {}
            exchanges.append((verb, headers, message))
            method = message.get("method")
            status, media = 200, "application/json"
            extra = ""
            if verb == "DELETE" and fault == "hang_delete":
                await reader.read()
                return
            if verb == "GET" and resume:
                media, payload = (
                    "text/event-stream",
                    b"data: " + resume["payload"] + b"\n\n",
                )
            elif verb == "DELETE" or "id" not in message or "method" not in message:
                status, payload = 202, b""
            else:
                if method == "initialize":
                    result = {
                        "protocolVersion": version,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "owned", "version": "1"},
                    }
                    extra = "Mcp-Session-Id: owned-session\r\n"
                elif method == "server/discover":
                    result = {
                        "supportedVersions": [version],
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "owned", "version": "1"},
                    }
                elif method == "tools/list":
                    result = {
                        "tools": [
                            {
                                "name": "echo",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "region": {
                                            "type": "string",
                                            "x-mcp-header": "Region",
                                        }
                                    },
                                },
                            }
                        ]
                    }
                    if fault == "cursor":
                        result["nextCursor"] = "repeat"
                    if fault == "empty_cursor" and "cursor" not in message["params"]:
                        result["nextCursor"] = ""
                    if fault == "invalid_annotation":
                        result["tools"].append(
                            {
                                "name": "bad",
                                "inputSchema": {
                                    "type": "object",
                                    "properties": {
                                        "x": {"type": "number", "x-mcp-header": "Bad"}
                                    },
                                },
                            }
                        )
                elif method == "tools/call":
                    calls["count"] += 1
                    result = {
                        "content": [{"type": "text", "text": "ok"}],
                        "structuredContent": {"pass": True},
                        "_meta": {"source": "owned"},
                    }
                    if fault == "lost":
                        return
                    if fault == "hang":
                        await reader.read()
                        return
                else:
                    result = {method.split("/")[0]: []}
                payload = json.dumps(
                    {"jsonrpc": "2.0", "id": message["id"], "result": result}
                ).encode()
                if fault == "unsupported" and method == "server/discover":
                    status = 400
                    payload = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "error": {
                                "code": -32022,
                                "message": "Unsupported",
                                "data": {
                                    "supported": (
                                        supported
                                        if supported is not None
                                        else ["2099-01-01", "2025-11-25"]
                                    )
                                },
                            },
                        }
                    ).encode()
                if fault == "unadvertised" and method in (
                    "resources/list",
                    "prompts/list",
                ):
                    payload = json.dumps(
                        {
                            "jsonrpc": "2.0",
                            "id": message["id"],
                            "error": {
                                "code": -32601,
                                "message": "Method not supported",
                            },
                        }
                    ).encode()
                if fault == "auth":
                    status, payload = 401, b""
                if fault == "redirect":
                    status, extra, payload = (
                        307,
                        "Location: http://127.0.0.1:9/stolen\r\n",
                        b"private-redirect-body-sentinel",
                    )
                if fault == "cancel_connect":
                    await reader.read()
                    return
                if method == "tools/call" and fault == "duplicate":
                    payload = (
                        b'{"jsonrpc":"2.0","id":'
                        + str(message["id"]).encode()
                        + b',"result":{"structuredContent":{"pass":true,"pass":false}}}'
                    )
                if method == "tools/call" and fault == "oversize":
                    payload = b" " * 1_048_577
                if method == "tools/call":
                    if fault == "expiry":
                        status, payload = 404, b""
                    elif fault == "wrong_id":
                        payload = json.dumps(
                            {"jsonrpc": "2.0", "id": 999999, "result": result}
                        ).encode()
                    elif fault == "deep":
                        payload = (
                            b'{"jsonrpc":"2.0","id":'
                            + str(message["id"]).encode()
                            + b',"result":{"x":'
                            + b"[" * 65
                            + b"0"
                            + b"]" * 65
                            + b"}}"
                        )
                    elif fault == "rpc_error":
                        payload = json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": message["id"],
                                "error": {
                                    "code": -32602,
                                    "message": "private sentinel",
                                },
                            }
                        ).encode()
                    elif fault == "input_required":
                        payload = json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": message["id"],
                                "result": {
                                    "resultType": "input_required",
                                    "inputRequests": [],
                                },
                            }
                        ).encode()
                if mode in ("sse", "sse_cr") and status == 200:
                    media = "text/event-stream"
                    if fault == "batch" and method == "tools/call":
                        payload = b"[" + payload + b"]"
                    payload = b": keepalive\r\n\r\ndata: " + payload + b"\r\n\r\n"
                    if method == "tools/call":
                        if fault == "duplicate_final":
                            payload += payload
                        elif fault == "malformed_event":
                            payload = b"data: broken\n\n"
                        elif fault == "server_request":
                            payload = (
                                b'data: {"jsonrpc":"2.0","id":"server-request","method":"sampling/createMessage","params":{}}\n\n'
                                + payload
                            )
                        elif fault == "list_changed":
                            payload = (
                                b'data: {"jsonrpc":"2.0","method":"notifications/tools/list_changed"}\n\n'
                                + payload
                            )
                        elif fault == "resume":
                            resume["payload"] = payload.split(b"data: ", 1)[1].strip()
                            payload = b"id: event-1\ndata:\nretry: 10\n\n"
                if mode == "sse_cr":
                    payload = payload.replace(b"\r\n", b"\r").replace(b"\n", b"\r")
                if fault == "compressed" and method == "tools/call":
                    import gzip

                    payload = gzip.compress(b" " * 2_000_000)
                    extra = "Content-Encoding: gzip\r\n"
                if fault == "bad_media" and method == "tools/call":
                    media = "text/html"
            framing = f"Content-Length: {len(payload)}\r\nConnection: close\r\n"
            if fault == "keepalive" and method == "tools/call":
                framing = (
                    f"Content-Length: {len(payload)}\r\nConnection: keep-alive\r\n"
                )
            if fault == "open_final" and method == "tools/call":
                framing = "Connection: keep-alive\r\n"
            writer.write(
                f"HTTP/1.1 {status} OK\r\nContent-Type: {media}; charset=utf-8\r\n{extra}{framing}\r\n".encode()
                + payload
            )
            await writer.drain()
            if fault in {"open_final", "keepalive"} and method == "tools/call":
                await reader.read()
        finally:
            writers.discard(writer)
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    old_connect = socket.socket.connect
    old_dns = socket.getaddrinfo

    def connect(sock, address):
        if sock.family == socket.AF_INET and address == ("127.0.0.1", port):
            return _socket.socket.connect(sock, address)
        return old_connect(sock, address)

    def dns(host, service, *args, **kwargs):
        if host in ("127.0.0.1", b"127.0.0.1") and int(service) == port:
            return _socket.getaddrinfo(host, service, *args, **kwargs)
        return old_dns(host, service, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(socket.socket, "connect", connect)
        scoped.setattr(socket, "getaddrinfo", dns)
        try:
            yield f"http://127.0.0.1:{port}/mcp", exchanges, calls
        finally:
            server.close()
            await server.wait_closed()
            for writer in list(writers):
                writer.close()
            for task in list(tasks):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("mode", ("json", "sse", "sse_cr"))
async def test_actual_profile_discovery_and_complete_result(monkeypatch, version, mode):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, version=version, mode=mode) as (url, messages, calls):
        client = MCPClient()
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    protocol_version=version,
                    development_loopback=True,
                )
            )
            assert calls["count"] == 0
            result = await client.call_tool_result(
                "owned", "echo", {"region": " 世界 "}
            )
            assert result.structured_content == {"pass": True}
            assert result.content[0]["text"] == "ok"
            assert result.duplicate_keys_checked
            assert result.encoded_payload
            assert result.dispatch_state == "settled"
            posted = [entry for entry in messages if entry[0] == "POST"]
            first = posted[0][2]["method"]
            assert first == (
                "server/discover" if version == VERSIONS[0] else "initialize"
            )
            for _, headers, message in posted:
                if version == VERSIONS[0]:
                    assert headers["mcp-protocol-version"] == version
                    assert headers["mcp-method"] == message["method"]
                    assert (
                        message["params"]["_meta"][
                            "io.modelcontextprotocol/protocolVersion"
                        ]
                        == version
                    )
                    assert "mcp-session-id" not in headers
                elif message["method"] != "initialize":
                    assert headers["mcp-session-id"] == "owned-session"
                    assert headers.get("mcp-protocol-version") == (
                        version if version == VERSIONS[1] else None
                    )
            if version == VERSIONS[0]:
                assert posted[-1][1]["mcp-param-region"] == "=?base64?IOS4lueVjCA=?="
        finally:
            await client.disconnect_all()
        assert not client.sessions and not client._pending_connections


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("cursor", "auth", "redirect"))
async def test_failed_discovery_never_publishes_ready(monkeypatch, fault):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, fault=fault) as (url, _messages, calls):
        client = MCPClient()
        assert not await client.connect_profile(
            TransportProfile(
                profile_id="owned",
                transport="streamable_http",
                url=url,
                development_loopback=True,
            )
        )
        assert not client.sessions and not client._pending_connections
        assert calls["count"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ("lost", "duplicate", "oversize"))
@pytest.mark.parametrize("mode", ("json", "sse"))
async def test_invalid_or_lost_call_stays_uncertain_without_replay(
    monkeypatch, fault, mode
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, fault=fault, mode=mode) as (url, _messages, calls):
        client = MCPClient()
        profile = TransportProfile(
            profile_id="owned",
            transport="streamable_http",
            url=url,
            development_loopback=True,
        )
        try:
            assert await client.connect_profile(profile)
            result = await client.call_tool_result("owned", "echo", {})
            assert result.transport_error
            assert result.dispatch_state == "uncertain"
            assert calls["count"] == 1
            assert await client.connect_profile(profile)
            assert calls["count"] == 1
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fault",
    (
        "wrong_id",
        "deep",
        "duplicate_final",
        "malformed_event",
        "bad_media",
        "input_required",
        "compressed",
    ),
)
async def test_protocol_failures_retire_readiness_and_remain_uncertain(
    monkeypatch, fault
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, fault=fault, mode="sse") as (url, _, calls):
        client = MCPClient()
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                )
            )
            result = await client.call_tool_result("owned", "echo", {})
            assert result.transport_error
            assert result.dispatch_state == "uncertain"
            assert not client.sessions
            assert calls["count"] == 1
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_server_request_and_batch_rules_are_profile_specific(
    monkeypatch, version
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    for fault in ("server_request", "batch"):
        async with peer(monkeypatch, version=version, fault=fault, mode="sse") as (
            url,
            messages,
            calls,
        ):
            client = MCPClient()
            try:
                assert await client.connect_profile(
                    TransportProfile(
                        profile_id="owned",
                        transport="streamable_http",
                        url=url,
                        protocol_version=version,
                        development_loopback=True,
                    )
                )
                result = await client.call_tool_result("owned", "echo", {})
                success = (
                    version != VERSIONS[0]
                    if fault == "server_request"
                    else version == VERSIONS[2]
                )
                assert (result.transport_error is None) == success
                if success:
                    assert result.dispatch_state == "settled"
                    assert result.structured_content == {"pass": True}
                if fault == "server_request" and success:
                    replies = [
                        message
                        for _, _, message in messages
                        if message.get("id") == "server-request"
                    ]
                    assert replies == [
                        {
                            "jsonrpc": "2.0",
                            "id": "server-request",
                            "error": {
                                "code": -32601,
                                "message": "Method not supported",
                            },
                        }
                    ]
                assert calls["count"] == 1
            finally:
                await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS[1:])
async def test_legacy_resume_and_session_expiry_do_not_replay_call(
    monkeypatch, version
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    for fault in ("resume", "expiry"):
        async with peer(monkeypatch, version=version, fault=fault, mode="sse") as (
            url,
            messages,
            calls,
        ):
            client = MCPClient()
            profile = TransportProfile(
                profile_id="owned",
                transport="streamable_http",
                url=url,
                protocol_version=version,
                development_loopback=True,
            )
            try:
                assert await client.connect_profile(profile)
                result = await client.call_tool_result("owned", "echo", {})
                assert calls["count"] == 1
                if fault == "resume":
                    assert result.structured_content == {"pass": True}
                    gets = [headers for verb, headers, _ in messages if verb == "GET"]
                    assert gets[0]["last-event-id"] == "event-1"
                    assert gets[0]["mcp-session-id"] == "owned-session"
                else:
                    assert (
                        result.dispatch_state == "uncertain" and result.transport_error
                    )
                    assert not client.sessions
                assert await client.connect_profile(profile)
                assert calls["count"] == 1
            finally:
                await client.disconnect_all()


@pytest.mark.asyncio
async def test_empty_cursor_and_invalid_header_tool_have_working_siblings(monkeypatch):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    for fault in ("empty_cursor", "invalid_annotation"):
        async with peer(monkeypatch, fault=fault) as (url, messages, _):
            client = MCPClient()
            try:
                assert await client.connect_profile(
                    TransportProfile(
                        profile_id="owned",
                        transport="streamable_http",
                        url=url,
                        development_loopback=True,
                    )
                )
                assert all(
                    tool["name"] == "echo" for tool in client.get_server_tools("owned")
                )
                assert (
                    await client.call_tool_result("owned", "echo", {})
                ).structured_content == {"pass": True}
                if fault == "empty_cursor":
                    lists = [
                        message
                        for _, _, message in messages
                        if message.get("method") == "tools/list"
                    ]
                    assert len(lists) == 2 and lists[1]["params"]["cursor"] == ""
            finally:
                await client.disconnect_all()


@pytest.mark.asyncio
async def test_cancelled_connect_releases_pending_connection(monkeypatch):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, fault="cancel_connect") as (url, messages, _):
        client = MCPClient()
        task = asyncio.create_task(
            client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                )
            )
        )
        async with asyncio.timeout(2):
            while not messages:
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert (
            not client.sessions
            and not client._pending_connections
            and not client._connect_reservations
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_call_cancellation_has_real_write_observation_and_no_replay(
    monkeypatch, version
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile
    from tldw_chatbook.MCP.tool_results import MCPDispatchObservation, observe_dispatch

    async with peer(monkeypatch, version=version, fault="hang") as (
        url,
        messages,
        calls,
    ):
        client = MCPClient()
        profile = TransportProfile(
            profile_id="owned",
            transport="streamable_http",
            url=url,
            protocol_version=version,
            development_loopback=True,
        )
        try:
            assert await client.connect_profile(profile)
            observation = MCPDispatchObservation()
            with observe_dispatch(observation):
                task = asyncio.create_task(client.call_tool_result("owned", "echo", {}))
            async with asyncio.timeout(2):
                while calls["count"] == 0:
                    await asyncio.sleep(0.001)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert observation.state == "uncertain"
            methods = [message.get("method") for _, _, message in messages]
            assert ("notifications/cancelled" in methods) == (version != VERSIONS[0])
            assert await client.connect_profile(profile)
            assert calls["count"] == 1
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_list_change_retires_old_definitions_until_reconnect(
    monkeypatch, version
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, version=version, mode="sse", fault="list_changed") as (
        url,
        _,
        calls,
    ):
        client = MCPClient()
        profile = TransportProfile(
            profile_id="owned",
            transport="streamable_http",
            url=url,
            protocol_version=version,
            development_loopback=True,
        )
        try:
            assert await client.connect_profile(profile)
            first = await client.call_tool_result("owned", "echo", {})
            assert first.dispatch_state == "settled"
            assert not client.sessions
            assert await client.connect_profile(profile)
            assert calls["count"] == 1
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("supported", [["2099-01-01"], ["2025-11-25"], ["2026-07-28"]])
async def test_modern_unsupported_version_never_sends_legacy_handshake(
    monkeypatch, supported
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, fault="unsupported", supported=supported) as (
        url,
        messages,
        _,
    ):
        client = MCPClient()
        assert not await client.connect_profile(
            TransportProfile(
                profile_id="owned",
                transport="streamable_http",
                url=url,
                development_loopback=True,
            )
        )
        assert client.connection_diagnostics["owned"] == "mcp_protocol_unsupported"
        assert [m.get("method") for _, _, m in messages] == ["server/discover"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested, offered", [("2025-03-26", "2025-11-25"), ("2025-11-25", "2025-03-26")]
)
async def test_legacy_counteroffer_selects_whole_qualified_profile(
    monkeypatch, requested, offered
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, version=offered) as (url, messages, _):
        client = MCPClient()
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    protocol_version=requested,
                    development_loopback=True,
                )
            )
            assert client.sessions["owned"].protocol_version == offered
            assert (
                await client.call_tool_result("owned", "echo", {})
            ).dispatch_state == "settled"
            for _, headers, message in messages:
                if message.get("method") != "initialize":
                    assert headers.get("mcp-protocol-version") == (
                        offered if offered == "2025-11-25" else None
                    )
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
async def test_invalid_header_argument_is_not_started_and_does_not_kill_control(
    monkeypatch,
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch) as (url, _, calls):
        client = MCPClient()
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                )
            )
            result = await client.call_tool_result(
                "owned", "echo", {"region": {"bad": True}}
            )
            assert result.dispatch_state == "not_started" and result.transport_error
            assert calls["count"] == 0
            assert (
                await client.call_tool_result("owned", "echo", {"region": "ok"})
            ).dispatch_state == "settled"
            assert calls["count"] == 1
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
async def test_final_sse_result_closes_stream_without_waiting_for_peer_eof(monkeypatch):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, fault="open_final", mode="sse") as (url, _, calls):
        client = MCPClient()
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                )
            )
            result = await asyncio.wait_for(
                client.call_tool_result("owned", "echo", {}), timeout=0.25
            )
            assert result.dispatch_state == "settled" and result.structured_content == {
                "pass": True
            }
            assert calls["count"] == 1
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_optional_unadvertised_catalogs_do_not_block_tools(monkeypatch, version):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, version=version, fault="unadvertised") as (
        url,
        messages,
        _,
    ):
        client = MCPClient()
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    protocol_version=version,
                    development_loopback=True,
                )
            )
            assert (
                await client.call_tool_result("owned", "echo", {})
            ).dispatch_state == "settled"
            assert not any(
                m.get("method") in ("resources/list", "prompts/list")
                for _, _, m in messages
            )
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("offered", ["2026-07-28", "2099-01-01"])
@pytest.mark.parametrize("requested", VERSIONS[1:])
async def test_legacy_refuses_modern_or_unknown_counteroffer(
    monkeypatch, requested, offered
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, version=offered) as (url, messages, calls):
        client = MCPClient()
        assert not await client.connect_profile(
            TransportProfile(
                profile_id="owned",
                transport="streamable_http",
                url=url,
                protocol_version=requested,
                development_loopback=True,
            )
        )
        assert not client.sessions and not client._pending_connections
        assert calls["count"] == 0
        assert not any(
            m.get("method") in ("notifications/initialized", "server/discover")
            for _, _, m in messages
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize(
    "query",
    [
        "",
        "?route=alpha&tag=one&tag=two&empty=&encoded=a%2Fb%26c%3Dd",
        "?route=${M2_ENDPOINT_ROUTE}&literal=%24%7BM2_ENDPOINT_ROUTE%7D",
    ],
)
async def test_literal_query_reaches_post_and_legacy_resume_delete(
    monkeypatch, version, query
):
    from urllib.parse import unquote

    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    monkeypatch.setenv("M2_ENDPOINT_ROUTE", "host-value-must-not-appear")
    legacy = version != VERSIONS[0]
    async with peer(
        monkeypatch, version=version, mode="sse", fault="resume" if legacy else None
    ) as (url, messages, calls):
        client = MCPClient()
        try:
            profile = TransportProfile(
                profile_id="owned",
                transport="streamable_http",
                url=url + query,
                protocol_version=version,
                development_loopback=True,
            )
            assert await client.connect_profile(profile)
            result = await client.call_tool_result("owned", "echo", {})
            assert result.dispatch_state == "settled"
            assert result.structured_content == {"pass": True}
            assert result.duplicate_keys_checked
        finally:
            await client.disconnect_all()
        assert calls["count"] == 1
        assert {verb for verb, _ in calls["targets"]} == (
            {"POST", "GET", "DELETE"} if legacy else {"POST"}
        )
        for _verb, target in calls["targets"]:
            # Only the normal HTTP encoder may escape literal braces; no semantic
            # parsing/reordering of repeated, empty or encoded query values.
            assert unquote(target) == unquote("/mcp" + query)
            if "${" not in query:
                assert target == "/mcp" + query
            assert "host-value-must-not-appear" not in target
        assert all("authorization" not in headers for _, headers, _ in messages)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_query_endpoint_redirect_refusal_has_same_entry_success(
    monkeypatch, version
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    query = "?route=alpha&tag=one&tag=two&empty="
    for fault in (None, "redirect"):
        async with peer(monkeypatch, version=version, fault=fault) as (
            url,
            _messages,
            calls,
        ):
            client = MCPClient()
            try:
                connected = await client.connect_profile(
                    TransportProfile(
                        profile_id="owned",
                        transport="streamable_http",
                        url=url + query,
                        protocol_version=version,
                        development_loopback=True,
                    )
                )
                assert connected == (fault is None)
                if connected:
                    assert (
                        await client.call_tool_result("owned", "echo", {})
                    ).dispatch_state == "settled"
                    assert calls["count"] == 1
                else:
                    assert (
                        client.connection_diagnostics["owned"] == "mcp_redirect_refused"
                    )
                    assert "private-redirect-body-sentinel" not in str(
                        client.connection_diagnostics
                    )
                    assert calls["targets"] == [("POST", "/mcp" + query)]
                    assert not client.sessions and not client._pending_connections
            finally:
                await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
@pytest.mark.parametrize("fault", [None, "hang"])
async def test_disconnect_closes_http_resources_before_caller_finalization(
    monkeypatch, version, fault
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile
    from tldw_chatbook.MCP.tool_results import MCPDispatchObservation, observe_dispatch

    async with peer(monkeypatch, version=version, fault=fault) as (url, _, calls):
        client = MCPClient()
        release = asyncio.Event()
        finalizing = asyncio.Event()
        caller = None
        session = None
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    protocol_version=version,
                    development_loopback=True,
                )
            )
            session = client.sessions["owned"]
            observation = MCPDispatchObservation()

            async def call_and_finalize():
                try:
                    return await client.call_tool_result("owned", "echo", {})
                except asyncio.CancelledError:
                    finalizing.set()
                    await release.wait()
                    raise

            with observe_dispatch(observation):
                caller = asyncio.create_task(call_and_finalize())
            async with asyncio.timeout(2):
                while not calls["count"]:
                    await asyncio.sleep(0.001)
            connections = list(session._http._transport._pool.connections)
            if fault is None:
                result = await caller
                assert result.structured_content == {"pass": True}
            await asyncio.wait_for(client.disconnect_all(), timeout=8)
            assert session._http.is_closed
            assert not session._http._transport._pool.connections
            assert all(connection.is_closed() for connection in connections)
            if fault == "hang":
                assert finalizing.is_set() and not caller.done()
                assert observation.state == "uncertain"
            else:
                assert observation.state == "settled"
            assert not client.sessions and not client._pending_connections
            await session.close()
            assert session._http.is_closed and calls["count"] == 1
        finally:
            release.set()
            if caller is not None:
                await asyncio.gather(caller, return_exceptions=True)
            await client.disconnect_all()
            if session is not None:
                await session._http.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS[1:])
async def test_interrupted_http_close_finishes_resources_and_can_repeat(
    monkeypatch, version
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, version=version, fault="hang_delete") as (
        url,
        messages,
        calls,
    ):
        client = MCPClient()
        session = None
        closing = None
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    protocol_version=version,
                    development_loopback=True,
                )
            )
            session = client.sessions["owned"]
            assert (
                await client.call_tool_result("owned", "echo", {})
            ).dispatch_state == "settled"
            closing = asyncio.create_task(session.close())
            async with asyncio.timeout(2):
                while not any(verb == "DELETE" for verb, _, _ in messages):
                    await asyncio.sleep(0.001)
            connections = list(session._http._transport._pool.connections)
            with pytest.raises(RuntimeError, match="mcp_connection_closed"):
                await session.notify("notifications/cancelled", {"requestId": 999})
            closing.cancel()
            with pytest.raises(asyncio.CancelledError):
                await closing
            await asyncio.wait_for(session.close(), timeout=2)
            assert session._http.is_closed
            assert all(connection.is_closed() for connection in connections)
            await client.disconnect_all()
            assert not client.sessions and calls["count"] == 1
        finally:
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            await client.disconnect_all()
            if session is not None:
                await session._http.aclose()


@pytest.mark.asyncio
async def test_stalled_real_http_close_keeps_bounded_retryable_owner(
    monkeypatch, tmp_path
):
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
    from tldw_chatbook.MCP.local_store import (
        LocalExternalMCPProfile,
        LocalMCPStore,
        TransportProfile,
    )

    async with peer(monkeypatch) as (url, _, calls):
        client = MCPClient()
        release = asyncio.Event()
        closing_http = asyncio.Event()
        session = None
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                )
            )
            session = client.sessions["owned"]
            assert (
                await client.call_tool_result("owned", "echo", {})
            ).dispatch_state == "settled"
            real_close = session._http.aclose

            async def delayed_real_close():
                closing_http.set()
                await release.wait()
                await real_close()

            monkeypatch.setattr(session._http, "aclose", delayed_real_close)
            assert not await asyncio.wait_for(
                client.disconnect_from_server("owned"), timeout=7
            )
            store = LocalMCPStore(tmp_path / "closing.json")
            store.save_profile(
                LocalExternalMCPProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                    protocol_version="2026-07-28",
                )
            )
            local = LocalMCPControlService(
                store=store, client=client, manifest_provider=dict
            )
            assert closing_http.is_set() and not session._http.is_closed
            assert local.get_external_servers()[0]["is_connected"] is False
            assert client.get_server_tools("owned") == []
            before = len(calls["targets"])
            assert not await asyncio.wait_for(
                client.connect_profile(
                    TransportProfile(
                        profile_id="owned",
                        transport="streamable_http",
                        url=url,
                        development_loopback=True,
                    )
                ),
                timeout=7,
            )
            assert len(calls["targets"]) == before
            assert client.sessions["owned"] is session
            with pytest.raises(RuntimeError, match="mcp_connection_closed"):
                await session.request("tools/list")
            retry = asyncio.create_task(session.close())
            await asyncio.sleep(0)
            retry.cancel()
            with pytest.raises(asyncio.CancelledError):
                await retry
            assert client.sessions["owned"] is session
            release.set()
            await asyncio.wait_for(client.disconnect_all(), timeout=2)
            assert session._http.is_closed
            assert not session._http._transport._pool.connections
            assert not client.sessions and calls["count"] == 1
            await session.close()
            assert await local.connect_profile("owned")
            assert local.get_external_servers()[0]["is_connected"] is True
            assert client.sessions["owned"] is not session
        finally:
            release.set()
            await client.disconnect_all()
            if session is not None:
                await session._http.aclose()


@pytest.mark.asyncio
async def test_failed_lower_http_close_never_settles_from_client_closed_bit(
    monkeypatch,
):
    import time

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, fault="keepalive") as (url, _, calls):
        client = MCPClient()
        session = None
        real_close = None
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                )
            )
            session = client.sessions["owned"]
            assert (
                await client.call_tool_result("owned", "echo", {})
            ).dispatch_state == "settled"
            connections = list(session._http._transport._pool.connections)
            assert connections and any(
                not connection.is_closed() for connection in connections
            )
            real_close = session._http._transport.aclose

            async def failed_lower_close():
                raise RuntimeError("owned lower-close failure")

            monkeypatch.setattr(session._http._transport, "aclose", failed_lower_close)
            assert not await client.disconnect_from_server("owned")
            assert session._http.is_closed
            assert any(not connection.is_closed() for connection in connections)
            assert client.sessions["owned"] is session
            # HTTPX's second aclose is a no-op after it sets its CLOSED bit.
            assert not await client.disconnect_from_server("owned")
            assert client.sessions["owned"] is session
            assert any(not connection.is_closed() for connection in connections)
            assert calls["count"] == 1
            client._maintenance_close_admission()
            assert not await client._maintenance_drain(time.monotonic() + 0.2)
            with pytest.raises(RecoveryRequired, match="runtime_work_not_settled"):
                client._maintenance_resume()
        finally:
            if real_close is not None:
                await real_close()
            client._producer_lifetime.resume()
            await client.disconnect_all()
            if session is not None:
                assert not session._http._transport._pool.connections


@pytest.mark.asyncio
@pytest.mark.parametrize("version", VERSIONS)
async def test_actual_http_maintenance_retires_pool_and_requires_fresh_connect(
    monkeypatch, version
):
    import time

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, version=version) as (url, _, calls):
        client = MCPClient()
        profile = TransportProfile(
            profile_id="owned",
            transport="streamable_http",
            url=url,
            protocol_version=version,
            development_loopback=True,
        )
        session = None
        try:
            assert await client.connect_profile(profile)
            session = client.sessions["owned"]
            assert (
                await client.call_tool_result("owned", "echo", {})
            ).dispatch_state == "settled"
            client._maintenance_close_admission()
            session._producer_lifetime.close()
            before = len(calls["targets"])
            with pytest.raises(RecoveryRequired, match="runtime_producer_paused"):
                await session.request("tools/call", {"name": "echo", "arguments": {}})
            assert len(calls["targets"]) == before
            assert await client._maintenance_drain(time.monotonic() + 3)
            assert (
                session._cleanup_complete
                and not session._http._transport._pool.connections
            )
            assert client._maintenance_children_exited()
            client._maintenance_resume()
            assert not client.sessions and calls["count"] == 1
            assert await client.connect_profile(profile)
            assert client.sessions["owned"] is not session
            assert calls["count"] == 1
        finally:
            if session is not None:
                await session.close()
            client._producer_lifetime.resume()
            await client.disconnect_all()


@pytest.mark.asyncio
async def test_http_maintenance_rejoins_actual_stalled_pool_close(monkeypatch):
    import time

    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch) as (url, _, calls):
        client = MCPClient()
        release, entered = asyncio.Event(), asyncio.Event()
        session = None
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                )
            )
            session = client.sessions["owned"]
            real_close = session._http.aclose

            async def delayed_close():
                entered.set()
                await release.wait()
                await real_close()

            monkeypatch.setattr(session._http, "aclose", delayed_close)
            client._maintenance_close_admission()
            assert not await client._maintenance_drain(time.monotonic() + 0.1)
            await asyncio.wait_for(entered.wait(), 2)
            retained = session._cleanup_task
            assert client.sessions["owned"] is session and not session._cleanup_complete
            assert not client.get_server_tools("owned")
            with pytest.raises(RecoveryRequired, match="runtime_work_not_settled"):
                client._maintenance_resume()
            release.set()
            assert await client._maintenance_drain(time.monotonic() + 3)
            assert session._cleanup_task is retained and session._cleanup_complete
            client._maintenance_resume()
            assert not client.sessions and calls["count"] == 0
        finally:
            release.set()
            if session is not None:
                await session.close()
            client._producer_lifetime.resume()
            await client.disconnect_all()


@pytest.mark.asyncio
async def test_raw_http_request_retains_actual_storage_admission(monkeypatch):
    from tldw_chatbook.Backup_Recovery import storage_admission
    from tldw_chatbook.MCP.activation import MCPActivationRequired
    from tldw_chatbook.MCP.client import MCPClient
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch) as (url, _, calls):
        client = MCPClient()
        try:
            assert await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                )
            )
            session = client.sessions["owned"]
            before = len(calls["targets"])
            pause = storage_admission._begin_local_pause()
            try:
                with pytest.raises(
                    MCPActivationRequired, match="mcp_activation_required"
                ):
                    await session.request(
                        "tools/call", {"name": "echo", "arguments": {}}
                    )
            finally:
                pause.resume()
            assert len(calls["targets"]) == before
            assert (
                await client.call_tool_result("owned", "echo", {})
            ).dispatch_state == "settled"
            assert calls["count"] == 1
        finally:
            await client.disconnect_all()
