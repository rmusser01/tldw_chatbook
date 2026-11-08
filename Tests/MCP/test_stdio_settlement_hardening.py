"""MCP settlement retains custody; proven unsent refusals keep a live session."""

from __future__ import annotations

import asyncio

import pytest

from Tests.MCP.test_client_catalog_pagination import _bare_connection
from tldw_chatbook.MCP import client as client_module


class _UnreapableStdin:
    def close(self) -> None:
        return None

    async def wait_closed(self) -> None:
        return None


class _UnreapableProcess:
    """A child that reports exit only when this fixture explicitly releases it."""

    def __init__(self) -> None:
        self.stdin = _UnreapableStdin()
        self.stdout = None
        self.stderr = None
        self.returncode = None
        self.exited = asyncio.Event()
        self.kill_attempted = asyncio.Event()
        self.terminate_calls = 0
        self.kill_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1
        self.kill_attempted.set()

    def finish(self) -> None:
        self.returncode = -1
        self.exited.set()

    async def wait(self) -> int:
        await self.exited.wait()
        return self.returncode


def _settled_bare_connection(process=None):
    connection = _bare_connection(process or _UnreapableProcess())
    connection._close_lock = asyncio.Lock()
    return connection


@pytest.mark.asyncio
async def test_failed_settlement_retains_custody_until_child_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(
        client_module, "_SETTLE_EXIT_GRACE_SECONDS", 0.05, raising=False
    )
    connection = _settled_bare_connection()
    process = connection.process
    waiter = asyncio.create_task(connection._settle_failed_request())
    try:
        await asyncio.wait_for(process.kill_attempted.wait(), 2.0)
        await asyncio.sleep(0.25)
        assert process.terminate_calls >= 1 and process.kill_calls >= 1
        assert process.returncode is None
        assert not waiter.done(), "settlement returned while native work remained"
        process.finish()
        await asyncio.wait_for(waiter, 2.0)
    finally:
        process.finish()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_shared_request_retains_custody_until_child_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        client_module, "_SETTLE_EXIT_GRACE_SECONDS", 0.05, raising=False
    )
    connection = _settled_bare_connection()
    future = asyncio.get_running_loop().create_future()
    future.cancel()
    waiter = asyncio.create_task(connection._retain_shared_request(future))
    try:
        await asyncio.sleep(0.15)
        assert connection.process.returncode is None
        assert not waiter.done(), "shared custody returned before native exit"
        connection.process.finish()
        with pytest.raises(RuntimeError, match="MCP transport unavailable"):
            await asyncio.wait_for(waiter, 2.0)
    finally:
        connection.process.finish()
        await asyncio.gather(waiter, return_exceptions=True)


@pytest.mark.asyncio
async def test_unsent_request_failure_does_not_tear_down_the_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failures before the frame left (params guard, newline guard) keep the session."""
    connection = _settled_bare_connection()

    async def refusing_send(_payload, **_kwargs):
        raise ValueError("MCP JSON-RPC messages must not contain embedded newlines")

    monkeypatch.setattr(connection, "_send_message", refusing_send)

    with pytest.raises(ValueError, match="embedded newlines"):
        await connection.request("tools/call", {"name": "tool", "arguments": {}})

    assert not connection._reader_unavailable
    assert not connection._cleanup_complete
    assert getattr(connection, "_request_settlement", None) is None
    assert connection._deadline_transport_healthy()

    # The session is still usable: a well-formed request can be sent and
    # answered on the same connection (the stub writer plays the reply).
    sent_payloads: list[dict] = []

    async def accepting_send(payload, **_kwargs):
        sent_payloads.append(payload)
        future = connection._pending_requests.get(payload["id"])
        if future is not None and not future.done():
            future.set_result({"content": [{"type": "text", "text": "ok"}]})

    monkeypatch.setattr(connection, "_send_message", accepting_send)
    result = await asyncio.wait_for(
        connection.request("tools/call", {"name": "tool", "arguments": {}}),
        0.5,
    )
    assert sent_payloads and sent_payloads[0]["method"] == "tools/call"
    assert result["content"][0]["text"] == "ok"
    assert not connection._reader_unavailable
    assert getattr(connection, "_request_settlement", None) is None


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", [ValueError, LookupError])
async def test_cleanup_exception_keeps_custody_until_child_exit(error_type):
    connection = _settled_bare_connection()
    entered = asyncio.Event()

    async def fail_cleanup():
        entered.set()
        raise error_type("controlled cleanup failure")

    connection._on_transport_failure = fail_cleanup
    waiter = asyncio.create_task(connection._settle_failed_request())
    try:
        await asyncio.wait_for(entered.wait(), 2.0)
        await asyncio.sleep(0.1)
        assert connection.process.returncode is None
        assert not waiter.done(), "cleanup failure released a live child"
        connection.process.finish()
        with pytest.raises(error_type, match="controlled cleanup failure"):
            await asyncio.wait_for(waiter, 2.0)
    finally:
        connection.process.finish()
        await asyncio.gather(waiter, return_exceptions=True)
