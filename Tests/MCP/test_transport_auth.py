"""Actual selected-origin HTTP requests use the current host binding secrets."""

import json

import pytest

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.requires_cleanup]

from Tests.MCP.test_credential_bindings import MemoryCredentialBackend
from Tests.MCP.test_streamable_http import peer
from tldw_chatbook.MCP.client import MCPClient
from tldw_chatbook.MCP.credential_bindings import (
    CredentialBindingService,
    HostCredential,
    endpoint_origin,
)
from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
from tldw_chatbook.MCP.local_store import LocalMCPStore


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["json", "sse"])
async def test_live_connection_resolves_renewed_token_and_refuses_changed_account(
    tmp_path, monkeypatch, mode
):
    async with peer(monkeypatch, mode=mode) as (url, messages, calls):
        identity = {"principal": "first", "token": "secret-sentinel-first"}

        async def adapter(reference):
            return HostCredential(
                "bearer",
                endpoint_origin(url),
                {"Authorization": "Bearer " + identity["token"]},
                issuer="issuer",
                audience="audience",
                principal=identity["principal"],
                scopes=("read",),
            )

        credentials = CredentialBindingService(
            MemoryCredentialBackend(), adapters={"host": adapter}
        )
        binding = await credentials.create("host")
        client = MCPClient()
        store = LocalMCPStore(tmp_path / "mcp.json")
        owner = LocalMCPControlService(
            store=store, client=client, credential_service=credentials
        )
        owner.save_external_profile(
            {
                "profile_id": "owned",
                "transport": "streamable_http",
                "protocol_version": "2026-07-28",
                "url": url,
                "development_loopback": True,
                "credential_reference": binding.reference_id,
                "credential_generation": binding.authority_generation,
            }
        )
        try:
            assert (await owner.connect_profile("owned"))["tools"][0]["name"] == "echo"
            identity["token"] = "secret-sentinel-renewed"
            await credentials.renew(binding.reference_id)
            result = await client.call_tool_result("owned", "echo", {})
            assert result.structured_content == {"pass": True}
            assert result.metadata == {"source": "owned"}
            assert messages[-1][1]["authorization"] == "Bearer secret-sentinel-renewed"
            assert calls["count"] == 1
            identity["principal"] = "second"
            await credentials.renew(binding.reference_id)
            refused = await client.call_tool_result("owned", "echo", {})
            assert refused.transport_error is not None
            assert calls["count"] == 1
            assert "secret-sentinel" not in repr(
                (binding, result, refused, owner.get_external_servers())
            )
            assert "secret-sentinel" not in store.path.read_text()
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", ["redirect", "auth", "lost"])
async def test_credential_failure_never_redirects_or_replays(
    tmp_path, monkeypatch, fault
):
    async with peer(monkeypatch, fault=fault) as (url, messages, calls):
        credentials = CredentialBindingService(MemoryCredentialBackend())
        binding = credentials.create_opaque(
            endpoint_origin=endpoint_origin(url),
            headers={"Authorization": "Bearer secret-sentinel"},
            method="bearer",
        )
        client = MCPClient(credential_service=credentials)
        from tldw_chatbook.MCP.local_store import TransportProfile

        try:
            connected = await client.connect_profile(
                TransportProfile(
                    profile_id="owned",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                    credential_reference=binding.reference_id,
                    credential_generation=binding.authority_generation,
                )
            )
            if fault == "lost":
                assert connected
                result = await client.call_tool_result("owned", "echo", {})
                assert result.transport_error is not None
                assert calls["count"] == 1
            else:
                assert not connected
                assert len(messages) == 1
            assert "secret-sentinel" not in json.dumps(client.connection_diagnostics)
            assert all(
                headers["authorization"] == "Bearer secret-sentinel"
                for _, headers, _ in messages
            )
        finally:
            await client.disconnect_all()


@pytest.mark.asyncio
async def test_latin1_octets_reach_owned_socket_without_protocol_override(monkeypatch):
    import _socket
    import asyncio
    import socket

    from tldw_chatbook.MCP.local_store import TransportProfile
    from tldw_chatbook.MCP.streamable_http import StreamableHTTPConnection

    received = []
    tasks = set()

    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            header = await reader.readuntil(b"\r\n\r\n")
            received.append(header)
            length = next(
                int(line.split(b":", 1)[1])
                for line in header.split(b"\r\n")
                if line.lower().startswith(b"content-length:")
            )
            request = json.loads(await reader.readexactly(length))
            payload = json.dumps(
                {"jsonrpc": "2.0", "id": request["id"], "result": {}}
            ).encode()
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: "
                + str(len(payload)).encode()
                + b"\r\nConnection: close\r\n\r\n"
                + payload
            )
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    previous = socket.socket.connect

    def connect(sock, address):
        if sock.family == socket.AF_INET and address == ("127.0.0.1", port):
            return _socket.socket.connect(sock, address)
        return previous(sock, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    url = f"http://127.0.0.1:{port}/rpc"
    credentials = CredentialBindingService(MemoryCredentialBackend())
    binding = credentials.create_opaque(
        endpoint_origin=endpoint_origin(url),
        headers={"X-Key": "secret-sentinel-é", "X-Empty": "", "X-Interior": "a\tb"},
    )
    connection = StreamableHTTPConnection(
        TransportProfile(
            profile_id="octets",
            transport="streamable_http",
            url=url,
            development_loopback=True,
            credential_reference=binding.reference_id,
            credential_generation=binding.authority_generation,
        ),
        credential_service=credentials,
    )
    try:
        await connection.request("ping")
        assert len(received) == 1
        assert b"x-key: secret-sentinel-\xe9\r\n" in received[0]
        assert b"x-empty: \r\n" in received[0]
        assert b"x-interior: a\tb\r\n" in received[0]
        assert b"Mcp-Method: ping\r\n" in received[0]
        assert b"MCP-Protocol-Version: 2026-07-28\r\n" in received[0]
    finally:
        await connection.close()
        server.close()
        await server.wait_closed()
        await asyncio.gather(*tasks)


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["revoke", "expire"])
async def test_legacy_notification_and_delete_refusal_still_close_resources(
    monkeypatch, change
):
    from tldw_chatbook.MCP.credential_bindings import CredentialError
    from tldw_chatbook.MCP.local_store import TransportProfile

    async with peer(monkeypatch, version="2025-11-25") as (url, messages, _calls):
        credentials = CredentialBindingService(MemoryCredentialBackend())
        binding = credentials.create_opaque(
            endpoint_origin=endpoint_origin(url), headers={"X-Key": "secret-sentinel"}
        )
        client = MCPClient(credential_service=credentials)
        assert await client.connect_profile(
            TransportProfile(
                profile_id="legacy",
                transport="streamable_http",
                protocol_version="2025-11-25",
                url=url,
                development_loopback=True,
                credential_reference=binding.reference_id,
                credential_generation=binding.authority_generation,
            )
        )
        session = client.sessions["legacy"]
        before = len(messages)
        if change == "revoke":
            credentials.revoke(binding.reference_id)
        else:
            record = json.loads(credentials.backend.records[binding.reference_id])
            record["expires_at"] = 1
            credentials.backend.records[binding.reference_id] = json.dumps(record)
        with pytest.raises(CredentialError):
            await session.notify("notifications/cancelled", {"requestId": 1})
        await client.disconnect_all()
        assert session._cleanup_complete
        assert session._http.is_closed
        assert not session._requests
        assert len(messages) == before


@pytest.mark.asyncio
async def test_expired_token_fails_before_discovery_despite_saved_profile(
    tmp_path, monkeypatch
):
    async with peer(monkeypatch) as (url, messages, _calls):
        credentials = CredentialBindingService(MemoryCredentialBackend())
        binding = credentials.create_opaque(
            endpoint_origin=endpoint_origin(url),
            headers={"X-Key": "secret-sentinel"},
            expires_at=1,
        )
        client = MCPClient(credential_service=credentials)
        from tldw_chatbook.MCP.local_store import TransportProfile

        assert not await client.connect_profile(
            TransportProfile(
                profile_id="expired",
                transport="streamable_http",
                url=url,
                development_loopback=True,
                credential_reference=binding.reference_id,
                credential_generation=binding.authority_generation,
            )
        )
        assert client.connection_diagnostics["expired"] == "credential_expired"
        assert not messages
        assert not client.sessions
        assert not client._pending_connections


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_blocked_credential_backend_does_not_block_peer_or_emit_after_timeout(
    monkeypatch,
    cancel,
):
    import asyncio
    import threading
    import time

    from tldw_chatbook.MCP.local_store import TransportProfile
    from tldw_chatbook.MCP.streamable_http import StreamableHTTPConnection

    async with peer(monkeypatch) as (url, messages, _calls):
        backend = MemoryCredentialBackend()
        credentials = CredentialBindingService(backend)
        binding = credentials.create_opaque(
            endpoint_origin=endpoint_origin(url), headers={"X-Key": "secret-sentinel"}
        )
        started, release = threading.Event(), threading.Event()
        original = backend.read

        def blocked(reference):
            started.set()
            release.wait(3)
            return original(reference)

        backend.read = blocked
        timer = threading.Timer(0.4, release.set)
        timer.start()
        auth = StreamableHTTPConnection(
            TransportProfile(
                profile_id="auth",
                transport="streamable_http",
                url=url,
                development_loopback=True,
                credential_reference=binding.reference_id,
                credential_generation=binding.authority_generation,
            ),
            credential_service=credentials,
        )
        ordinary = StreamableHTTPConnection(
            TransportProfile(
                profile_id="ordinary",
                transport="streamable_http",
                url=url,
                development_loopback=True,
            )
        )
        try:
            before = time.monotonic()
            pending = asyncio.create_task(auth.request("ping", timeout_seconds=0.03))
            await asyncio.sleep(0.01)
            assert started.is_set()
            await ordinary.request("ping")
            assert time.monotonic() - before < 0.2
            if cancel:
                pending.cancel()
            with pytest.raises(asyncio.CancelledError if cancel else TimeoutError):
                await pending
            # A second lookup cannot queue behind the retained blocked owner.
            with pytest.raises(TimeoutError):
                await auth.request("ping", timeout_seconds=0.03)
            assert len(messages) == 1
            release.set()
            async with asyncio.timeout(1):
                while credentials._worker_future is not None:
                    await asyncio.sleep(0.001)
            assert len(messages) == 1
            await auth.request("ping")
            assert len(messages) == 2
            assert messages[-1][1]["x-key"] == "secret-sentinel"
        finally:
            release.set()
            timer.cancel()
            await auth.close()
            await ordinary.close()
            await asyncio.gather(pending, return_exceptions=True)


@pytest.mark.asyncio
async def test_short_contention_waits_then_resolves_each_callers_own_secret(
    monkeypatch,
):
    import asyncio
    import time

    from tldw_chatbook.MCP.local_store import TransportProfile
    from tldw_chatbook.MCP.streamable_http import StreamableHTTPConnection

    async with peer(monkeypatch) as (url, messages, _calls):
        backend = MemoryCredentialBackend()
        credentials = CredentialBindingService(backend)
        bindings = [
            credentials.create_opaque(
                endpoint_origin=endpoint_origin(url),
                headers={"X-Key": f"secret-sentinel-{index}"},
            )
            for index in range(2)
        ]
        original = backend.read
        reads = []

        def slow(reference):
            reads.append(reference)
            time.sleep(0.03)
            return original(reference)

        backend.read = slow
        connections = [
            StreamableHTTPConnection(
                TransportProfile(
                    profile_id=f"owned{index}",
                    transport="streamable_http",
                    url=url,
                    development_loopback=True,
                    credential_reference=binding.reference_id,
                    credential_generation=binding.authority_generation,
                ),
                credential_service=credentials,
            )
            for index, binding in enumerate(bindings)
        ]
        try:
            await asyncio.gather(
                *(connection.request("ping") for connection in connections)
            )
            assert reads == [binding.reference_id for binding in bindings]
            assert [headers["x-key"] for _, headers, _ in messages] == [
                "secret-sentinel-0",
                "secret-sentinel-1",
            ]
        finally:
            await asyncio.gather(*(connection.close() for connection in connections))


@pytest.mark.asyncio
async def test_changed_origin_refuses_before_emitting_headers(monkeypatch):
    from tldw_chatbook.MCP.credential_bindings import CredentialError
    from tldw_chatbook.MCP.local_store import TransportProfile
    from tldw_chatbook.MCP.streamable_http import StreamableHTTPConnection

    async with peer(monkeypatch) as (url, messages, _calls):
        credentials = CredentialBindingService(MemoryCredentialBackend())
        binding = credentials.create_opaque(
            endpoint_origin="https://different.example",
            headers={"X-Key": "secret-sentinel"},
        )
        connection = StreamableHTTPConnection(
            TransportProfile(
                profile_id="wrong-origin",
                transport="streamable_http",
                url=url,
                development_loopback=True,
                credential_reference=binding.reference_id,
                credential_generation=binding.authority_generation,
            ),
            credential_service=credentials,
        )
        try:
            with pytest.raises(CredentialError, match="credential_changed"):
                await connection.request("ping")
            assert not messages
        finally:
            await connection.close()


@pytest.mark.asyncio
async def test_credential_worker_start_failure_is_sanitized_and_releases_capacity(
    monkeypatch,
):
    import threading

    from tldw_chatbook.MCP.credential_bindings import CredentialError
    from tldw_chatbook.MCP.local_store import TransportProfile
    from tldw_chatbook.MCP.streamable_http import StreamableHTTPConnection

    async with peer(monkeypatch) as (url, messages, _calls):
        credentials = CredentialBindingService(MemoryCredentialBackend())
        binding = credentials.create_opaque(
            endpoint_origin=endpoint_origin(url), headers={"X-Key": "secret-sentinel"}
        )
        connection = StreamableHTTPConnection(
            TransportProfile(
                profile_id="owned",
                transport="streamable_http",
                url=url,
                development_loopback=True,
                credential_reference=binding.reference_id,
                credential_generation=binding.authority_generation,
            ),
            credential_service=credentials,
        )
        try:
            with monkeypatch.context() as patch:

                def failed_start(thread):
                    raise RuntimeError("secret-sentinel-thread-failure")

                patch.setattr(threading.Thread, "start", failed_start)
                with pytest.raises(
                    CredentialError, match="credential_storage_unavailable"
                ):
                    await connection.request("ping")
                assert credentials._worker_future is None and not messages
            await connection.request("ping")
            assert len(messages) == 1
        finally:
            await connection.close()
