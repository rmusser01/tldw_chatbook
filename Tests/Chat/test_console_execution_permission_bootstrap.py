"""Actual pre-mount permission mutation and adoption of its one backend owner."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time

import pytest

from Tests.private_profile import private_profile_test


@private_profile_test
@pytest.mark.asyncio
@pytest.mark.parametrize("ambient_loop", [True, False], ids=["ambient-loop", "no-loop"])
async def test_original_permission_bootstrap_retires_before_async_start(request, ambient_loop):
    from Tests.Chat.test_console_control_trace import _observe
    from Tests.Chat.test_console_execution_backend_process import _ObservedStderr
    from Tests.Chat.test_console_execution_profile import _settled_profile
    from tldw_chatbook.Chat.console_execution_permission_service import ConsoleProcessPermissionService
    from tldw_chatbook.Chat.console_execution_process import (
        ConsoleChannelLost,
        ConsoleExecutionClient,
        ConsoleProcessProfile,
    )
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    path, directory, *_ = _settled_profile()
    # 'ask' is the stock default. Start with a different actual persisted value
    # so returning a default or merely queueing a mutation cannot pass.
    original_permissions = MCPPermissionStore(directory / "mcp_permissions.json")
    original_permissions.set_global_default("deny")
    assert original_permissions.get_global_default() == "deny"
    profile = ConsoleProcessProfile(str(path), str(directory))
    client = ConsoleExecutionClient(None)
    service = ConsoleProcessPermissionService(client, profile)
    epochs = client.profile_epoch, client.service_epoch
    processes, stderr_tail = [], bytearray()
    closed = False

    def created(frame):
        caller = frame.f_back
        if (caller.f_code is ConsoleExecutionClient._launch.__code__
            and caller.f_locals.get("self") is client):
            process = frame.f_locals["self"]
            assert type(process) is subprocess.Popen and process.stderr is not None
            processes.append(process)
            process.stderr = _ObservedStderr(process.stderr, stderr_tail)

    def set_without_loop():
        with pytest.raises(RuntimeError, match="no running event loop"):
            asyncio.get_running_loop()
        return service.set_global_default("ask")

    try:
        assert client._process is None and client._start_task is None
        with _observe(subprocess.Popen.__init__.__code__, sys.monitoring.events.PY_RETURN, created):
            if ambient_loop:
                assert asyncio.get_running_loop() is not None
                result = service.set_global_default("ask")
            else:
                result = await asyncio.to_thread(set_without_loop)
            assert result is None
            bootstrap = client._permission_bootstrap
            assert bootstrap is not None and bootstrap.receipt is not None
            assert bootstrap.receipt.received.done() and bootstrap.receipt.received.result() is None
            assert bootstrap.receipt.applied.done()
            assert bootstrap.receipt.applied.result() == {"kind": "SetGlobalDefault", "result": None}
            assert bootstrap.receipt.retired.done() and bootstrap.receipt.retired.result() is True
            assert bootstrap.thread is not None and bootstrap.thread.ident is not None
            assert not bootstrap.thread.is_alive()
            # This real strict read precedes client.start: the synchronous
            # operation must already have applied its original mutation.
            before_start = original_permissions.read_snapshot_strict()
            assert before_start.payload["profiles"]["default"]["global_default"] == "ask"
            process, tree = client._process, client._tree
            assert process is not None and process.pid != os.getpid()
            assert process.poll() is None and tree is not None
            assert processes == [process]
            assert (client.profile_epoch, client.service_epoch) == epochs

            await client.start(profile)
            assert client._loop is asyncio.get_running_loop()
            assert client._ready.done() and client._ready.exception() is None
            assert client._process is process and client._tree is tree
            assert processes == [process]
            assert (client.profile_epoch, client.service_epoch) == epochs

            await client.start(profile)
            assert client._process is process and client._tree is tree
            assert processes == [process]
            assert (client.profile_epoch, client.service_epoch) == epochs
            with pytest.raises(ConsoleChannelLost):
                service.set_global_default("deny")

        read = client.submit("PermissionAction", {
            "operation": {"kind": "ReadPermissionSnapshot", "strict": True},
        })
        await asyncio.shield(read.received)
        value = await asyncio.shield(read.applied)
        assert await asyncio.shield(read.retired) is True
        assert value["kind"] == "ReadPermissionSnapshot"
        snapshot = value["result"]
        assert snapshot["kind"] == "snapshot" and snapshot["file_exists"] is True
        assert snapshot["payload"]["profiles"]["default"]["global_default"] == "ask"
        assert client._process is process and client._tree is tree
        assert (client.profile_epoch, client.service_epoch) == epochs
    finally:
        try:
            closed = await client.shutdown(deadline=time.monotonic() + 30)
        finally:
            tail = stderr_tail.decode("utf-8", errors="replace")
            request.node.user_properties.append(("actual_backend_stderr_tail", tail))
            if tail and not closed:
                print("Actual backend diagnostic tail (bounded):", tail, file=sys.stderr)

    assert closed is True and client._disposed and client.physically_closed
    assert process.returncode == 0
    assert all(stream.closed for stream in (process.stdin, process.stdout, process.stderr))
    assert not client._monitor.is_alive()
    assert all(not thread.is_alive() for thread in client._channel._started)
    request.node.user_properties.append(("actual_backend_pid", process.pid))
    request.node.user_properties.append(("bootstrap_ambient_loop", ambient_loop))
