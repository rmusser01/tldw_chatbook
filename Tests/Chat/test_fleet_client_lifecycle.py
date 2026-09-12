"""Pool ownership through real primary and surviving-child lifelines."""

import asyncio
import threading

import httpx
import pytest

from Tests.Agents.conftest import pin_agent_settings
from Tests.Chat.test_console_agent_bridge import (
    _await_event,
    _bridge_with_gateway,
    _CrossTurnGateway,
    _join_fleet_threads,
    _run,
)
from tldw_chatbook.Chat import console_agent_bridge
from tldw_chatbook.Chat.console_agent_bridge import _ModelCallLifeline
from tldw_chatbook.Chat.console_provider_gateway import ConsoleProviderGateway


@pytest.mark.parametrize("injected", [False, True])
def test_surviving_child_releases_its_pool_only_when_its_lifeline_ends(
    tmp_path, monkeypatch, injected
):
    pin_agent_settings(monkeypatch, subagents_outlive_turn=True)
    release_child = threading.Event()
    source = _CrossTurnGateway(release_child)
    external_client = httpx.AsyncClient() if injected else None
    gateway = ConsoleProviderGateway(http_client=external_client)
    clients = {}

    async def stream(resolution, messages, **kwargs):
        loop = asyncio.get_running_loop()
        clients[loop] = gateway._active_http_client()
        async for chunk in source.stream_chat(resolution, messages, **kwargs):
            yield chunk

    monkeypatch.setattr(gateway, "stream_chat", stream)
    bridge, db, store, session, aid = _bridge_with_gateway(tmp_path, gateway)
    try:
        outcome = _run(bridge, store, session, aid)
        assert outcome.status == "done"
        parent_loop = source.parent_loops[0]
        child_loop = source.child_loops[0]
        assert parent_loop.is_closed()
        assert not child_loop.is_closed()
        child_client = clients[child_loop]
        assert not child_client.is_closed
        # App teardown must not close a pool underneath the surviving call.
        asyncio.run(gateway.aclose())
        assert not child_client.is_closed
        release_child.set()
        _join_fleet_threads(timeout=10)
        child_rows = [
            r for r in db.list_runs("conv-1") if r["agent_kind"] == "subagent"
        ]
        assert len(child_rows) == 1
        assert child_rows[0]["result"] == "child answer after the turn"
        assert child_loop.is_closed()
        if injected:
            assert child_client is external_client
            assert not external_client.is_closed
        else:
            assert child_client.is_closed, "child pool outlived its owning lifeline"
            assert clients[parent_loop].is_closed, "primary pool also needs cleanup"
            assert not gateway._loop_clients
    finally:
        release_child.set()
        _join_fleet_threads(timeout=10)
        # Real clients above have no network I/O; the failing baseline still
        # needs explicit test cleanup after its closed loops strand the pools.
        for client in set(clients.values()):
            if not client.is_closed:
                asyncio.run(client.aclose())


def test_lifeline_drains_cancelled_request_before_closing_pool_once():
    gateway = ConsoleProviderGateway()
    entered = threading.Event()
    order = []
    clients = []

    async def request():
        clients.append(gateway._active_http_client())
        entered.set()
        try:
            await asyncio.Future()
        finally:
            await asyncio.sleep(0)
            order.append("request settled")

    async def cleanup():
        assert asyncio.get_running_loop() is lifeline.loop
        assert order == ["request settled"]
        await gateway.aclose_current_loop()
        order.append("pool closed")

    lifeline = _ModelCallLifeline("test-drain", close_current_loop=cleanup)
    lifeline.start()
    try:
        asyncio.run_coroutine_threadsafe(request(), lifeline.loop)
        assert entered.wait(2)
    finally:
        lifeline.shutdown()
    lifeline.shutdown()
    assert order == ["request settled", "pool closed"]
    assert clients[0].is_closed
    assert lifeline.loop.is_closed()
    assert not gateway._loop_clients


def test_slow_cleanup_keeps_owning_loop_alive_then_closes_it(monkeypatch):
    gateway = ConsoleProviderGateway()
    release = threading.Event()
    entered = threading.Event()
    monkeypatch.setattr(console_agent_bridge, "_LOOP_THREAD_JOIN_SECONDS", 0.05)

    async def cleanup():
        entered.set()
        assert await _await_event(release, 3)
        await gateway.aclose_current_loop()

    async def touch():
        return gateway._active_http_client()

    lifeline = _ModelCallLifeline("test-delayed-close", close_current_loop=cleanup)
    lifeline.start()
    client = asyncio.run_coroutine_threadsafe(touch(), lifeline.loop).result(2)
    try:
        lifeline.shutdown()
        assert entered.wait(2)
        assert not client.is_closed
        assert not lifeline.loop.is_closed()
        # Repeated shutdown must not stop the loop while it runs cleanup.
        lifeline.shutdown()
    finally:
        release.set()
        lifeline._thread.join(3)
    assert client.is_closed
    assert lifeline.loop.is_closed()
    assert not gateway._loop_clients


def test_cleanup_failure_still_closes_the_lifeline():
    async def cleanup():
        raise RuntimeError("test cleanup failure")

    lifeline = _ModelCallLifeline("test-failed-cleanup", close_current_loop=cleanup)
    lifeline.start()
    lifeline.shutdown()
    assert lifeline.loop.is_closed()
    assert not lifeline._thread.is_alive()


def test_pool_is_closed_before_the_driver_loop_stops(monkeypatch):
    gateway = ConsoleProviderGateway()
    lifeline = _ModelCallLifeline(
        "test-no-idle-window", close_current_loop=gateway.aclose_current_loop
    )
    stopped_with_closed_pool = []
    run_forever = lifeline.loop.run_forever

    def observe_stop():
        run_forever()
        stopped_with_closed_pool.append(client.is_closed)

    async def touch():
        return gateway._active_http_client()

    monkeypatch.setattr(lifeline.loop, "run_forever", observe_stop)
    lifeline.start()
    client = asyncio.run_coroutine_threadsafe(touch(), lifeline.loop).result(2)
    lifeline.shutdown()
    assert stopped_with_closed_pool and all(stopped_with_closed_pool), (
        "an idle-loop window lets app teardown steal the client's cleanup"
    )


def test_failed_thread_start_closes_loop_without_calling_cleanup(monkeypatch):
    called = []

    async def cleanup():
        called.append(True)

    lifeline = _ModelCallLifeline("test-failed-start", close_current_loop=cleanup)

    def fail_start():
        raise RuntimeError("thread exhausted")

    monkeypatch.setattr(lifeline._thread, "start", fail_start)
    with pytest.raises(RuntimeError, match="thread exhausted"):
        lifeline.start()
    lifeline.shutdown()
    lifeline.shutdown()
    assert lifeline.loop.is_closed()
    assert called == []
