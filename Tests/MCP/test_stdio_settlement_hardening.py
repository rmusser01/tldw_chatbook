"""TASK-38611: settlement never waits unbounded and unsent failures don't settle."""

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
    """A child that ignores terminate/kill and never reports an exit."""

    def __init__(self) -> None:
        self.stdin = _UnreapableStdin()
        self.stdout = None
        self.stderr = None
        self.returncode = None  # never set: the D-state shape
        self.terminate_calls = 0
        self.kill_calls = 0

    def terminate(self) -> None:
        self.terminate_calls += 1

    def kill(self) -> None:
        self.kill_calls += 1

    async def wait(self) -> int:
        await asyncio.Future()
        return -1


def _settled_bare_connection(process=None):
    connection = _bare_connection(process or _UnreapableProcess())
    connection._close_lock = asyncio.Lock()
    return connection


@pytest.mark.asyncio
async def test_settle_returns_within_grace_when_child_is_unreapable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreapable child surfaces the original failure, not an endless wait."""
    monkeypatch.setattr(client_module, "_TERMINATE_TIMEOUT_SECONDS", 0.05)
    monkeypatch.setattr(client_module, "_SETTLE_EXIT_GRACE_SECONDS", 0.05)
    connection = _settled_bare_connection()
    process = connection.process

    # Must return (no raise) well inside the 2s bound; pre-fix this hangs.
    await asyncio.wait_for(connection._settle_failed_request(), 2.0)

    assert process.terminate_calls >= 1 and process.kill_calls >= 1
    assert process.returncode is None  # still unreaped, as designed


@pytest.mark.asyncio
async def test_shared_request_custody_wait_is_bounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_retain_shared_request gives up after the grace period."""
    monkeypatch.setattr(client_module, "_SETTLE_EXIT_GRACE_SECONDS", 0.05)
    connection = _settled_bare_connection()
    future = asyncio.get_running_loop().create_future()
    future.cancel()

    with pytest.raises(RuntimeError, match="MCP transport unavailable"):
        await asyncio.wait_for(connection._retain_shared_request(future), 2.0)

    assert connection.process.returncode is None


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
